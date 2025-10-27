# Standard library imports
import os
import json
import re
import shutil
from pathlib import Path
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Third-party imports
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    send_from_directory,
    abort,
)
import yaml
import pandas as pd
from werkzeug.utils import secure_filename

# Local imports
from core.test_scenario_generator import generate_from_scenario

# ====================================
# Configuration and Initialization
# ====================================

app = Flask(__name__, static_folder="static")
app.secret_key = "your-secret-key-here"  # Change this to a secure secret key

# File and Directory Configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "swagger_files")
SCENARIOS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "scenarios")
TEST_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "tests")

ALLOWED_EXTENSIONS = {"json", "yaml", "yml"}

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SCENARIOS_FOLDER, exist_ok=True)
os.makedirs(TEST_FOLDER, exist_ok=True)

# App Configuration
app.config.update(
    UPLOAD_FOLDER=UPLOAD_FOLDER,
    TEST_FOLDER=TEST_FOLDER,
    SCENARIOS_FOLDER=SCENARIOS_FOLDER,
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
)

# ====================================
# Utility Functions
# ====================================


def resolve_refs(spec, root=None, visited_refs=None):
    """Recursively resolve $refs in the OpenAPI/Swagger specification.

    Args:
        spec: The current spec to process
        root: The root of the document (for resolving relative refs)
        visited_refs: Set of visited references to detect circular references
    """
    if root is None:
        root = spec
        visited_refs = set()
    elif visited_refs is None:
        visited_refs = set()

    if isinstance(spec, dict):
        if "$ref" in spec:
            ref = spec["$ref"]

            # Check for circular reference
            if ref in visited_refs:
                return {"$ref": ref, "description": "[Circular reference detected]"}

            # Add to visited references
            visited_refs.add(ref)

            try:
                # Resolve the reference
                ref_path = ref.split("/")[1:]  # Remove the leading #
                ref_value = root
                for part in ref_path:
                    ref_value = ref_value[part]

                # Resolve the referenced value
                resolved = resolve_refs(ref_value, root, set(visited_refs))
                return resolved

            except (KeyError, TypeError, AttributeError) as e:
                return {"$ref": ref, "error": f"Failed to resolve: {str(e)}"}

        else:
            # Process all values in the dictionary
            result = {}
            for k, v in spec.items():
                result[k] = resolve_refs(v, root, set(visited_refs))
            return result

    elif isinstance(spec, list):
        return [resolve_refs(item, root, set(visited_refs)) for item in spec]

    return spec


def load_swagger_file(file_path):
    """Load and parse a Swagger/OpenAPI file with resolved references."""
    with open(file_path, "r", encoding="utf-8") as file:
        if file_path.endswith((".yaml", ".yml")):
            spec = yaml.safe_load(file)
        else:
            spec = json.load(file)
    return resolve_refs(spec)


def get_all_endpoints_with_specs():
    """Get all endpoints along with their full resolved specifications."""
    all_data = []

    for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
        if any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            try:
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                # Get the full resolved spec
                full_spec = load_swagger_file(file_path)
                # Extract endpoints
                endpoints = extract_endpoints(full_spec, filename)

                # Add the full spec to each endpoint
                for endpoint in endpoints:
                    endpoint["_full_spec"] = full_spec

                all_data.extend(endpoints)
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    return all_data


def clean_data(obj, seen=None):
    """Recursively clean data to remove circular references."""
    if seen is None:
        seen = set()

    # Handle primitive types
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # Handle circular references
    obj_id = id(obj)
    if obj_id in seen:
        return "[Circular Reference]"
    seen.add(obj_id)

    # Handle dictionaries
    if isinstance(obj, dict):
        return {k: clean_data(v, seen) for k, v in obj.items()}

    # Handle lists and tuples
    if isinstance(obj, (list, tuple)):
        return [clean_data(item, seen) for item in obj]
    # For any other type, convert to string
    return str(obj)


def convert_openapi_to_swagger(input_file, output_file):
    """
    Converts an OpenAPI 3.x JSON/YAML spec to Swagger 2.0 JSON.
    Requires 'api-spec-converter' (Node.js package) installed globally:
        npm install -g api-spec-converter
    """
    try:
        # Run converter and capture stdout
        command = [
            "api-spec-converter",
            "--from=openapi_3",
            "--to=swagger_2",
            input_file,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=True)

        # Verify output and save to file
        if result.stdout.strip():
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.stdout)
            print(f"✅ Successfully converted {input_file} → {output_file}")
        else:
            print("⚠️ Conversion succeeded but produced no output.")
            print("Check that your input file is valid OpenAPI 3 JSON/YAML.")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error during conversion: {e}")
        print("stderr:", e.stderr)
    except FileNotFoundError:
        print(
            "❌ Error: 'api-spec-converter' not found. Please install it globally via:"
        )
        print("   npm install -g api-spec-converter")
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def parse_swagger_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        if file_path.endswith((".yaml", ".yml")):
            return yaml.safe_load(file)
        else:
            return json.load(file)


def extract_endpoints(swagger_data, filename):
    endpoints = []
    if not isinstance(swagger_data, dict):
        return endpoints

    base_path = swagger_data.get("basePath", "")
    paths = swagger_data.get("paths", {})

    for path, methods in paths.items():
        for method, details in methods.items():
            if method.lower() in [
                "get",
                "post",
                "put",
                "delete",
                "patch",
                "head",
                "options",
            ]:
                endpoint = {
                    "id": f"{method.upper()}:{base_path}{path}",
                    "method": method.upper(),
                    "path": f"{base_path}{path}",
                    "description": details.get("description", ""),
                    "summary": details.get("summary", ""),
                    "parameters": details.get("parameters", []),
                    "responses": details.get("responses", {}),
                    "file": os.path.basename(filename),
                }
                endpoints.append(endpoint)

    return endpoints


def get_all_endpoints():
    all_endpoints = []

    for filename in os.listdir(app.config["UPLOAD_FOLDER"]):
        if any(filename.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
            try:
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                swagger_data = parse_swagger_file(file_path)
                all_endpoints.extend(extract_endpoints(swagger_data, filename))
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")

    return all_endpoints


# Add this function before the route definitions
def parse_jtl_file(jtl_path):
    """Parse JTL file and return performance metrics."""
    try:
        # Read JTL file (CSV format)
        df = pd.read_csv(jtl_path)

        # Calculate metrics
        total_requests = len(df)
        if total_requests == 0:
            return None

        # Calculate error rate
        error_count = df[df["success"] == False].shape[0]
        error_rate = (error_count / total_requests) * 100 if total_requests > 0 else 0

        # Calculate percentiles
        response_times = df["elapsed"].dropna()
        if len(response_times) == 0:
            return None

        time_span = (
            df["timeStamp"].max() - df["timeStamp"].min()
        ) / 1000  # Convert to seconds
        if time_span == 0:
            time_span = 1  # Prevent division by zero

        connect_times = (
            df["Connect"].dropna() if "Connect" in df.columns else pd.Series([0])
        )
        latency = df["Latency"].mean() if "Latency" in df.columns else 0
        response_size = df["bytes"].mean() if "bytes" in df.columns else 0

        timestamp_match = re.search(r'results_(\d{8}_\d{6})', str(jtl_path))
        config_metrics = {
            "rampup": 0,
            "duration": 0,
            "loop_count": 1
        }
        if timestamp_match:
            timestamp = timestamp_match.group(1)
            # Look for config file with matching timestamp
            config_file = Path(jtl_path).parent / f"results_config_{timestamp}.json"
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    if "test_parameters" in config_data:
                        config_metrics = {
                            "rampup": config_data["test_parameters"].get("rampup", 0),
                            "duration": config_data["test_parameters"].get("duration", 0),
                            "loop_count": config_data["test_parameters"].get("loop_count", 1)
                        }

        metrics = {
            "samples": total_requests,  # Add total number of samples
            "tps": total_requests / time_span if time_span > 0 else 0,
            "avg_response_time": response_times.mean(),
            "median_response_time": response_times.median(),
            "p90_response_time": response_times.quantile(0.90),
            "p95_response_time": response_times.quantile(0.95),
            "p99_response_time": response_times.quantile(0.99),
            "error_rate": error_rate,
            "active_threads": (
                df["allThreads"].mean() if "allThreads" in df.columns else 0
            ),
            "throughput": (
                (df["bytes"].sum() / 1024 / time_span)
                if "bytes" in df.columns and time_span > 0
                else 0
            ),
            "latency": latency,
            "timestamp": datetime.fromtimestamp(os.path.getmtime(jtl_path)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            "response_size": response_size,
            "connect_time": connect_times.mean(),
        }

        metrics["rampup"] = config_metrics["rampup"]
        metrics["duration"] = config_metrics["duration"]
        metrics["loop_count"] = config_metrics["loop_count"]

        return metrics

    except Exception as e:
        app.logger.error(f"Error parsing JTL file {jtl_path}: {str(e)}")
        return None


def get_scenarios():
    """Get all saved scenarios from the scenarios directory."""
    scenarios_dir = Path(app.config["SCENARIOS_FOLDER"])
    scenarios = []

    for scenario_file in scenarios_dir.glob("*.json"):
        try:
            with open(scenario_file, "r") as f:
                scenario_data = json.load(f)
                scenario_data["filename"] = scenario_file.name
                scenarios.append(scenario_data)
        except (json.JSONDecodeError, IOError) as e:
            app.logger.error(f"Error reading scenario file {scenario_file}: {str(e)}")

    # Sort by creation date, newest first
    return sorted(scenarios, key=lambda x: x.get("created_at", ""), reverse=True)


# -------- Services (per swagger json/yaml) --------
def get_services():
    services = []
    upload_dir = Path(app.config["UPLOAD_FOLDER"])
    if not upload_dir.exists():
        return services
    for p in sorted(upload_dir.glob("*")):
        if p.suffix.lower() not in {".json", ".yaml", ".yml"}:
            continue
        try:
            # read spec
            if p.suffix.lower() == ".json":
                with open(p, "r", encoding="utf-8") as f:
                    spec = json.load(f)
            else:
                with open(p, "r", encoding="utf-8") as f:
                    spec = yaml.safe_load(f)

            info = spec.get("info", {}) if isinstance(spec, dict) else {}
            title = info.get("title") or p.stem
            version = info.get("version") or ""
            base_path = (
                spec.get("basePath") or spec.get("servers", [{}])[0].get("url")
                if isinstance(spec, dict)
                else ""
            )
            paths = spec.get("paths", {}) if isinstance(spec, dict) else {}
            endpoint_count = 0
            for path_item in paths.values():
                if isinstance(path_item, dict):
                    endpoint_count += sum(
                        1
                        for k in path_item.keys()
                        if k.lower()
                        in ["get", "post", "put", "delete", "patch", "options", "head"]
                    )
            services.append(
                {
                    "name": title,
                    "filename": p.name,
                    "version": version,
                    "base_path": base_path or "",
                    "endpoints": endpoint_count,
                    "modified_at": datetime.fromtimestamp(
                        p.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        except Exception:
            # skip unreadable files
            continue
    return services


def _build_scenario_from_endpoint(service_filename: str, method: str, path: str):
    """Construct a scenario_data dict for a single endpoint."""
    spec_path = Path(app.config["UPLOAD_FOLDER"]) / service_filename
    if not spec_path.exists():
        raise FileNotFoundError(f"Service spec not found: {service_filename}")

    # Load spec
    if spec_path.suffix.lower() == ".json":
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    else:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

    info = spec.get("info", {}) if isinstance(spec, dict) else {}
    service_name = info.get("title") or spec_path.stem

    # Get the specific endpoint details
    path_item = spec.get("paths", {}).get(path, {})
    op = path_item.get(method.lower(), {})
    desc = op.get("summary") or op.get("description") or f"{method.upper()} {path}"

    return {
        "name": f"{service_name} - {method.upper()} {path}",
        "endpoints": [
            {
                "method": method.upper(),
                "path": path,
                "description": desc,
                "file": service_filename,
            }
        ],
        "auth": {},
    }


def _build_scenario_from_service(service_filename: str):
    """Construct a scenario_data dict from a single swagger/openapi file, listing all operations."""
    spec_path = Path(app.config["UPLOAD_FOLDER"]) / service_filename
    if not spec_path.exists():
        raise FileNotFoundError(f"Service spec not found: {service_filename}")
    # Load spec
    if spec_path.suffix.lower() == ".json":
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = json.load(f)
    else:
        with open(spec_path, "r", encoding="utf-8") as f:
            spec = yaml.safe_load(f)

    info = spec.get("info", {}) if isinstance(spec, dict) else {}
    name = info.get("title") or spec_path.stem
    endpoints = []
    paths = spec.get("paths", {}) if isinstance(spec, dict) else {}
    for path, path_item in paths.items():
        if not isinstance(path_item, dict):
            continue
        for method, op in path_item.items():
            if method.lower() not in [
                "get",
                "post",
                "put",
                "delete",
                "patch",
                "options",
                "head",
            ]:
                continue
            desc = ""
            if isinstance(op, dict):
                desc = op.get("summary") or op.get("description") or ""
            endpoints.append(
                {
                    "method": method.upper(),
                    "path": path,
                    "description": desc or "No description",
                    "file": service_filename,
                    "id": f"{service_filename}:{method}:{path}",  # Add unique ID for each endpoint
                }
            )

    return {
        "name": name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "endpoints": endpoints,
    }


# ====================================
# Route Handlers
# ====================================
@app.route("/")
def index():
    return redirect(url_for("upload"))


# Add this route after your existing routes
@app.route("/api/test-results/<folder_id>/jtl-metrics")
def get_jtl_metrics(folder_id):
    try:
        base_dir = Path(app.config["TEST_FOLDER"])
        test_results_dir = os.path.join(base_dir, folder_id)

        if not os.path.exists(test_results_dir):
            return jsonify({"error": "Test results not found"}), 404

        # Find all JTL files in the folder
        jtl_files = []
        for root, _, files in os.walk(test_results_dir):
            for file in files:
                if file.endswith(".jtl"):
                    jtl_files.append(os.path.join(root, file))

        # Get metrics for each JTL file
        metrics_list = []
        for jtl_file in sorted(jtl_files, key=os.path.getmtime, reverse=True)[
            :10
        ]:  # Get 10 most recent
            metrics = parse_jtl_file(jtl_file)
            if metrics:
                metrics["file_name"] = os.path.basename(jtl_file)
                metrics_list.append(metrics)

        return jsonify({"success": True, "metrics": metrics_list})

    except Exception as e:
        app.logger.error(f"Error getting JTL metrics: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/test-results")
def test_results():
    # Use the project root directory as the base
    test_results_dir = Path(app.config["TEST_FOLDER"])
    test_folders = []

    try:
        # Create test_results directory if it doesn't exist
        os.makedirs(test_results_dir, exist_ok=True)

        # Get all test result directories
        for item in os.listdir(test_results_dir):
            item_path = os.path.join(test_results_dir, item)
            if os.path.isdir(item_path) and not item.startswith(
                "."
            ):  # Skip hidden directories
                # Get modification time
                modified = datetime.fromtimestamp(os.path.getmtime(item_path)).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )

                test_folders.append({"id": item, "name": item, "modified": modified})

        # Sort by modification time, newest first
        test_folders.sort(key=lambda x: x["modified"], reverse=True)

    except Exception as e:
        app.logger.error(f"Error reading test results: {str(e)}")
        flash("Error loading test results", "danger")
        test_folders = []

    return render_template("test_results.html", test_folders=test_folders)


@app.route("/api/get-server-configs")
def get_server_configs():
    configs = {}
    uploads_dir = app.config["UPLOAD_FOLDER"]
    for filename in os.listdir(uploads_dir):
        if filename.lower().endswith((".json", ".yaml", ".yml")):
            file_path = os.path.join(uploads_dir, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    if filename.lower().endswith((".yaml", ".yml")):
                        spec = yaml.safe_load(f)
                    else:
                        spec = json.load(f)

                if "info" in spec and "x-server-config" in spec["info"]:
                    configs[filename] = spec["info"]["x-server-config"]
            except Exception as e:
                print(f"Error reading {filename}: {str(e)}")
                continue

    return jsonify(configs)


@app.route("/api/save-server-config", methods=["POST"])
def save_server_config():
    """Save server configuration for a specific API file"""
    try:
        data = request.json
        filename = data.get("filename")

        if not filename:
            return jsonify({"error": "Filename is required"}), 400

        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        # Read the existing spec
        with open(file_path, "r", encoding="utf-8") as f:
            if filename.lower().endswith((".yaml", ".yml")):
                spec = yaml.safe_load(f)
            else:
                spec = json.load(f)

        # Update server config in info section
        if "info" not in spec:
            spec["info"] = {}

        spec["info"]["x-server-config"] = {
            "protocol": data.get("protocol", "https"),
            "host": data.get("host", ""),
            "port": data.get("port"),
        }

        # Save back to file
        with open(file_path, "w", encoding="utf-8") as f:
            if filename.lower().endswith((".yaml", ".yml")):
                yaml.dump(spec, f, default_flow_style=False, allow_unicode=True)
            else:
                json.dump(spec, f, indent=2)

        return jsonify({"message": "Server configuration saved successfully"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        if "files" not in request.files:
            flash("No files were selected", "error")
            return redirect(request.url)

        files = request.files.getlist("files")

        if not files or all(file.filename == "" for file in files):
            flash("No files were selected", "error")
            return redirect(request.url)

        success_count = 0

        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)

                try:
                    # Read the file content
                    content = file.read().decode("utf-8")

                    # Parse the content based on file extension
                    if filename.lower().endswith((".yaml", ".yml")):
                        import yaml

                        spec = yaml.safe_load(content)
                    else:
                        spec = json.loads(content)

                    # Save the file (either original or converted)
                    with open(file_path, "w", encoding="utf-8") as f:
                        f.write(content)

                    # Check if it's OpenAPI 3.x and convert to 2.0
                    if "openapi" in spec and spec["openapi"].startswith("3."):
                        try:
                            # Convert OpenAPI 3.x to 2.0
                            convert_openapi_to_swagger(file_path, file_path)
                            flash(
                                f"Successfully converted {filename} from OpenAPI 3.x to 2.0",
                                "success",
                            )
                        except Exception as e:
                            flash(
                                f"Error converting {filename} from OpenAPI 3.x to 2.0: {str(e)}",
                                "error",
                            )
                            continue

                    success_count += 1

                except json.JSONDecodeError:
                    flash(f"Invalid JSON in {filename}", "error")
                except yaml.YAMLError:
                    flash(f"Invalid YAML in {filename}", "error")
                except Exception as e:
                    flash(f"Error processing {filename}: {str(e)}", "error")

        if success_count > 0:
            flash(f"Successfully uploaded {success_count} file(s)")
        else:
            flash(
                "No valid files were uploaded. Allowed file types are: "
                + ", ".join(ALLOWED_EXTENSIONS),
                "error",
            )

        return redirect(url_for("endpoints"))

    # GET request or failed POST - show the upload form
    files = [
        f
        for f in os.listdir(app.config["UPLOAD_FOLDER"])
        if os.path.isfile(os.path.join(app.config["UPLOAD_FOLDER"], f))
        and f.lower().endswith(tuple(ALLOWED_EXTENSIONS))
    ]
    return render_template("upload.html", files=files, active_page="upload")


@app.route("/endpoints")
def endpoints():
    all_endpoints = get_all_endpoints_with_specs()
    cleaned_endpoints = []
    for endpoint in all_endpoints:
        cleaned = clean_data(endpoint)
        cleaned_endpoints.append(cleaned)
    return render_template(
        "endpoints.html", endpoints=cleaned_endpoints, active_page="endpoints"
    )

@app.route("/api/endpoints/<path:filename>")
def get_endpoints_by_file(filename):
    """Get all endpoints for a specific service file."""
    
    try:
        # Ensure the filename is safe
        from werkzeug.utils import secure_filename
        
        filename = secure_filename(filename)
        if not filename:
            return jsonify({"error": "Invalid filename"}), 400
            
        # Build the full file path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Check if file exists
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
            
        # Parse the file and extract endpoints
        try:
            swagger_data = parse_swagger_file(file_path)
            endpoints = extract_endpoints(swagger_data, filename)
            return jsonify(clean_data(endpoints))
        except Exception as e:
            app.logger.error(f"Error parsing {filename}: {str(e)}")
            return jsonify({"error": f"Failed to parse file: {str(e)}"}), 500
            
    except Exception as e:
        app.logger.error(f"Error getting endpoints for {filename}: {str(e)}")
        return jsonify({"error": "Failed to get endpoints"}), 500


@app.route("/api/files/<path:filename>", methods=["DELETE"])
def delete_file(filename):
    try:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if os.path.exists(filepath):
            os.remove(filepath)
            return jsonify(
                {"success": True, "message": f"File {filename} deleted successfully"}
            )
        else:
            return jsonify({"success": False, "error": "File not found"}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/status")
def status():
    return jsonify(
        {
            "status": "online",
            "timestamp": datetime.utcnow().isoformat(),
            "file_count": len(
                [
                    f
                    for f in os.listdir(app.config["UPLOAD_FOLDER"])
                    if os.path.isfile(os.path.join(app.config["UPLOAD_FOLDER"], f))
                ]
            ),
        }
    )


@app.template_filter("tojson")
def to_json(value):
    return json.dumps(value, ensure_ascii=False)


@app.template_filter("datetimeformat")
def datetime_format(value, format="%Y-%m-%d %H:%M:%S"):
    if not value:
        return ""
    try:
        # Try parsing ISO format
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        else:
            dt = value

        # If timezone-naive, assume UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        # Convert to local time
        local_dt = dt.astimezone()
        return local_dt.strftime(format)
    except (ValueError, AttributeError) as e:
        app.logger.error(f"Error formatting datetime {value}: {str(e)}")
        return str(value)


@app.route("/scenarios")
def list_scenarios():
    """List all saved scenarios."""
    scenarios = get_scenarios()
    return render_template(
        "scenarios.html", scenarios=scenarios, active_page="scenarios"
    )


@app.route("/services")
def list_services():
    services = get_services()
    return render_template("services.html", services=services, active_page="services")


@app.route("/api/generate_test_scripts", methods=["POST"])
def generate_test_scripts():
    """Unified endpoint for test generation.

    Request body should contain one of:
    1. For single endpoint:
        {"type": "endpoint", "service": "filename", "method": "GET", "path": "/api/endpoint"}
    2. For service:
        {"type": "service", "service": "filename", "scenario_name": "optional_name"}
    3. For scenario:
        {"type": "scenario", "scenario_data": {...}}
    """
    data = request.get_json() or {}
    test_type = data.get("type")

    try:
        if test_type == "endpoint":
            # Handle single endpoint
            service = data.get("service")
            method = data.get("method")
            path = data.get("path")
            if not all([service, method, path]):
                return (
                    jsonify(
                        {
                            "error": "For type 'endpoint', 'service', 'method', and 'path' are required"
                        }
                    ),
                    400,
                )
            scenario_data = _build_scenario_from_endpoint(service, method, path)
            scenario_name = (
                f"{service}_{method.lower()}_{path.replace('/', '_').strip('_')}"
            )

        elif test_type == "service":
            # Handle service
            service = data.get("service")
            if not service:
                return (
                    jsonify({"error": "For type 'service', 'service' is required"}),
                    400,
                )
            scenario_data = _build_scenario_from_service(service)
            scenario_name = data.get(
                "scenario_name", scenario_data.get("name", service)
            )

        elif test_type == "scenario":
            # Handle full scenario
            scenario_name = data.get("scenario_name")
            if not scenario_name:
                return jsonify({"error": "Scenario name is required"}), 400
            # Get the scenario file
            scenario_file = Path(app.config["SCENARIOS_FOLDER"]) / f"{scenario_name}.json"
            if not scenario_file.exists():
                return jsonify({"error": "Scenario not found"}), 404
            # Load scenario data
            with open(scenario_file, "r") as f:
                scenario_data = json.load(f)

        else:
            return (
                jsonify(
                    {
                        "error": "Invalid type. Must be one of: 'endpoint', 'service', 'scenario'"
                    }
                ),
                400,
            )

        # Apply UI configuration to scenario data if provided
        if 'config' in data:
            config = data['config']
            
            # Update test parameters
            if 'test_name' in config:
                scenario_name = config['test_name']
            
            # Update thread group settings
            if 'threads' in config and 'rampup' in config and 'duration' in config:
                if 'thread_group' not in scenario_data:
                    scenario_data['thread_group'] = {}
                scenario_data['thread_group'].update({
                    'threads': config['threads'],
                    'rampup': config['rampup'],
                    'duration': config['duration'],
                    'loopCount': config.get('loop_count', 1)
                })
            
            # Update endpoint weights if provided
            if 'endpoint_weights' in config and 'endpoints' in scenario_data:
                # Create a mapping of endpoint identifiers to their weights
                endpoint_weights = {
                    f"{e.get('method', '').upper()} {e.get('path', '')}": e.get('weight', 1)
                    for e in config['endpoint_weights']
                }
                
                # Update weights in the scenario data
                for endpoint in scenario_data['endpoints']:
                    endpoint_id = f"{endpoint.get('method', '').upper()} {endpoint.get('path', '')}"
                    if endpoint_id in endpoint_weights:
                        endpoint['weight'] = endpoint_weights[endpoint_id]
        
        # Common test generation logic
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_dir = Path(app.config["TEST_FOLDER"]) / scenario_name # / timestamp
        scenario_dir.mkdir(parents=True, exist_ok=True)

        # Generate test files
        result = generate_from_scenario(
            scenario_data=scenario_data,
            uploads_dir=app.config["UPLOAD_FOLDER"],
            output_dir=str(scenario_dir),
        )

        # Save the configuration as JSON
        config_data = {
            "scenario_name": scenario_name,
            "config": data.get('config', {}),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "test_type": test_type
        }
        
        config_path = scenario_dir / "test_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        # Add config file to the result
        result_files = result.get("files", [])
        result_files.append(str(config_path))

        return jsonify(
            {
                "success": True,
                "message": f"Test files generated successfully for {test_type}",
                "files": result_files,
                "jmx": result.get("jmx"),
                "output_dir": str(scenario_dir),
                "config_path": str(config_path)
            }
        )

    except Exception as e:
        app.logger.error(f"Error generating tests: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/scenarios/<scenario_name>", methods=["DELETE"])
def delete_scenario_api(scenario_name: str):
    """Delete a scenario JSON by its base name via API."""
    try:
        safe_name = re.sub(r"[^\w\-]", "_", scenario_name).lower()
        scenarios_dir = Path(app.config["SCENARIOS_FOLDER"])
        file_path = scenarios_dir / f"{safe_name}.json"
        if not file_path.exists():
            return jsonify({"success": False, "error": "Scenario not found"}), 404
        file_path.unlink()
        return jsonify({"success": True, "message": f"Scenario {safe_name} deleted"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/db/tests/<path:filepath>")
def serve_test_file(filepath):
    """Serve files from the test directory.
    
    Args:
        filepath: Relative path to the file within the test directory
        
    Returns:
        The requested file or 404 if not found
    """
    try:
        # Ensure the requested path is within the test directory
        test_dir = Path(app.config["TEST_FOLDER"])
        file_path = (test_dir / filepath).resolve()
        
        # Security check: prevent directory traversal
        if not file_path.is_relative_to(test_dir):
            abort(403, "Access denied")
            
        # Check if file exists and is a file
        if not file_path.is_file():
            abort(404, "File not found")
            
        # Send the file
        return send_from_directory(
            test_dir,
            filepath,
            as_attachment=False
        )
        
    except Exception as e:
        app.logger.error(f"Error serving test file {filepath}: {str(e)}")
        abort(404, "File not found")


@app.route("/save_scenario", methods=["POST"])
def save_scenario():
    try:
        data = request.get_json()
        if not data or "name" not in data or "endpoints" not in data:
            return jsonify({"error": "Invalid request data"}), 400

        # Create scenarios directory if it doesn't exist
        scenarios_dir = Path(app.config["SCENARIOS_FOLDER"])

        # Create a safe filename from the scenario name
        safe_name = re.sub(r"[^\w\-]", "_", data["name"]).lower()
        scenario_file = scenarios_dir / f"{safe_name}.json"

        # Check if file already exists and add a timestamp if it does
        counter = 1
        original_safe_name = safe_name
        while scenario_file.exists():
            safe_name = f"{original_safe_name}_{counter}"
            scenario_file = scenarios_dir / f"{safe_name}.json"
            counter += 1

        # Prepare scenario data
        scenario_data = {
            "name": data["name"],
            "created_at": datetime.now().isoformat(),
            "endpoints": data["endpoints"],
        }

        # Save to file
        with open(scenario_file, "w") as f:
            json.dump(scenario_data, f, indent=2)

        return jsonify(
            {"message": "Scenario saved successfully", "file": str(scenario_file)}
        )

    except Exception as e:
        app.logger.error(f"Error saving scenario: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/cleanup_tests", methods=["POST"])
def cleanup_tests():
    try:
        generated_tests_dir = Path(app.config["TEST_FOLDER"])

        # Remove all contents of generated_tests directory
        if os.path.exists(generated_tests_dir):
            for item in os.listdir(generated_tests_dir):
                item_path = os.path.join(generated_tests_dir, item)
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                elif os.path.isdir(item_path):
                    shutil.rmtree(item_path)

            return jsonify(
                {"success": True, "message": "Cleaned up all generated test files"}
            )
        else:
            return jsonify(
                {"success": True, "message": "No generated tests directory found"}
            )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/test-data")
def test_data():
    """Display all test data organized by folder."""
    generated_tests_dir = Path(app.config["TEST_FOLDER"])
    test_data = {}

    if generated_tests_dir.exists():
        for folder in generated_tests_dir.iterdir():
            if folder.is_dir():
                test_data[folder.name] = []
                # Get all files in the folder, excluding JMX files
                for file_path in folder.rglob("*"):
                    if file_path.is_file() and not file_path.name.lower().endswith(
                        ".jmx"
                    ):
                        # Get relative path from the folder
                        rel_path = file_path.relative_to(folder)
                        test_data[folder.name].append(
                            {
                                "name": file_path.name,
                                "path": str(file_path),
                                "size": file_path.stat().st_size,
                                "modified": datetime.fromtimestamp(
                                    file_path.stat().st_mtime
                                ).strftime("%Y-%m-%d %H:%M:%S"),
                                "relative_path": str(rel_path),
                            }
                        )

    return render_template(
        "test_data.html", test_data=test_data, active_page="test_data"
    )


@app.route("/test-results/<folder_id>/view-jtl")
def view_jtl_file(folder_id):
    """Serve JTL file content for viewing or as JSON."""
    try:
        file_name = request.args.get("file")
        download = request.args.get("download", "").lower() == "true"
        as_json = request.args.get("json", "").lower() == "true"

        if not file_name:
            if as_json:
                return jsonify({"success": False, "error": "No file specified"}), 400
            return "No file specified", 400

        # Prevent directory traversal
        if ".." in file_name or file_name.startswith("/"):
            if as_json:
                return jsonify({"success": False, "error": "Invalid file path"}), 400
            return "Invalid file path", 400

        # First try the exact folder_id, then try 'combo' as fallback
        possible_paths = [Path(app.config["TEST_FOLDER"]) / folder_id / file_name]

        file_path = None
        for path in possible_paths:
            if os.path.isfile(path):
                file_path = path
                break

        if not file_path:
            error_msg = f"File {file_name} not found in {folder_id} or combo folders"
            app.logger.error(error_msg)
            if as_json:
                return jsonify({"success": False, "error": error_msg}), 404
            return error_msg, 404

        # Read file content
        try:
            with open(file_path, "r") as f:
                content = f.read()

            if as_json:
                # Return as JSON with additional metadata
                return jsonify(
                    {
                        "success": True,
                        "file_name": file_name,
                        "size": os.path.getsize(file_path),
                        "last_modified": os.path.getmtime(file_path),
                        "content": content,
                        "line_count": len(content.splitlines()),
                        "word_count": len(content.split()),
                        "character_count": len(content),
                    }
                )
            elif download:
                # Return as file download
                return send_file(
                    file_path,
                    as_attachment=True,
                    download_name=file_name,
                    mimetype="text/plain",
                )
            else:
                # Return as plain text with proper content type
                response = make_response(content)
                response.mimetype = "text/plain"
                return response

        except Exception as e:
            app.logger.error(f"Error reading JTL file: {str(e)}")
            if as_json:
                return (
                    jsonify(
                        {"success": False, "error": f"Error reading file: {str(e)}"}
                    ),
                    500,
                )
            return f"Error reading file: {str(e)}", 500

    except Exception as e:
        app.logger.error(f"Error in view_jtl_file: {str(e)}")
        if as_json:
            return jsonify({"success": False, "error": f"Server error: {str(e)}"}), 500
        return f"Server error: {str(e)}", 500


@app.route("/api/upload-jtl", methods=["POST"])
def upload_jtl():
    """Handle JTL file uploads and save them with timestamp-based naming."""
    try:
        # Check if the post request has the file part
        if "jtl_file" not in request.files:
            return jsonify({"success": False, "error": "No file part"}), 400

        file = request.files["jtl_file"]
        folder_name = request.form.get("folder")

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == "":
            return jsonify({"success": False, "error": "No selected file"}), 400

        if not folder_name:
            return jsonify({"success": False, "error": "No folder specified"}), 400

        # Validate file extension
        if not (file and file.filename.endswith(".jtl")):
            return (
                jsonify({"success": False, "error": "Only .jtl files are allowed"}),
                400,
            )

        # Create the target directory if it doesn't exist
        target_dir = os.path.join(Path(app.config["TEST_FOLDER"]), folder_name)

        # Generate timestamp for the filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results_{timestamp}.jtl"
        filepath = os.path.join(target_dir, filename)

        # Save the file
        file.save(filepath)

        # Save the test configuration
        config_data = {
            "test_parameters": {
                "rampup": int(request.form.get("rampup", 60)),
                "duration": int(request.form.get("duration", 300)),
                "loop_count": int(request.form.get("loop_count", 1)),
                "test_name": folder_name,
                "original_filename": file.filename,
                "uploaded_at": datetime.now(timezone.utc).isoformat(),
                "jtl_file": filename
            }
        }

        # Save the config file with the same timestamp
        config_filename = f"results_config_{timestamp}.json"
        config_path = os.path.join(target_dir, config_filename)

        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

        app.logger.info(f"Test configuration saved to {config_path}")

        app.logger.info(f"JTL file saved to {filepath}")
        return jsonify(
            {
                "success": True,
                "message": "File uploaded successfully",
                "filename": filename,
                "timestamp": timestamp,
            }
        )

    except Exception as e:
        app.logger.error(f"Error uploading JTL file: {str(e)}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/save_csv", methods=["POST"])
def save_csv():
    """Save the edited CSV content to the specified file."""
    try:
        data = request.get_json()
        file_path = request.args.get("path")
        content = data.get("content", "")

        if not file_path:
            return jsonify({"success": False, "error": "No file path provided"}), 400

        # Ensure the path is safe and within the generated_tests directory
        full_path = Path(app.config["TEST_FOLDER"]) / file_path
        full_path = full_path.resolve()

        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the content to the file
        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        return jsonify({"success": True})

    except Exception as e:
        app.logger.error(f"Error saving CSV file: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    # Create necessary directories if they don't exist
    app.run(debug=True, host="0.0.0.0", port=5000)
