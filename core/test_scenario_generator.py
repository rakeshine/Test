import json, csv, os, random
import xml.etree.ElementTree as ET
from xml.dom import minidom
from collections import defaultdict

# --------- CONFIG ---------
COMBO_FILE = "combo.json"
OUTPUT_DIR = "combo_test"
DEFAULTS = {
    "threads": "${__P(threads, 10)}",
    "rampup": "${__P(rampup, 10)}",
    "duration": "${__P(duration, 10)}",
    "loopCount": "${__P(loopCount, 1)}"
}
# --------------------------

def pretty_xml(element):
    xml_str = ET.tostring(element, 'utf-8')
    return minidom.parseString(xml_str).toprettyxml(indent="  ")

def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# -----------------------------------------------------
# AUTH DETECTION
# -----------------------------------------------------
def extract_security_definitions(swagger):
    security_defs = swagger.get("securityDefinitions", {})
    auth_vars = {}
    for name, sec in security_defs.items():
        t = sec.get("type")
        if t == "apiKey":
            key_name = sec.get("name", "api_key")
            auth_vars[key_name] = "REPLACE_WITH_KEY"
        elif t == "oauth2":
            auth_vars["auth_token"] = "REPLACE_WITH_TOKEN"
    return auth_vars

# -----------------------------------------------------

def infer_csv_columns(swagger, path, method):
    """Infer CSV columns and parameter metadata from swagger for given path/method.
    Returns (columns, param_meta) where param_meta is list of {name, in}.
    Also includes a 'body' column if a body parameter exists.
    """
    paths = swagger.get("paths", {})
    method_l = method.lower()
    # Normalize incoming path: strip query and basePath
    norm_path = path.split("?", 1)[0]
    base_path = swagger.get("basePath", "") or ""
    candidates = [norm_path]
    if base_path and norm_path.startswith(base_path):
        stripped = norm_path[len(base_path):] or "/"
        if not stripped.startswith("/"):
            stripped = "/" + stripped
        candidates.insert(0, stripped)

    op = None
    hit_path = None
    for cand in candidates:
        if cand in paths and method_l in paths[cand]:
            op = paths[cand][method_l]
            hit_path = cand
            break
    if op is None:
        return [], []

    params = op.get("parameters", [])
    cols = []
    param_meta = []
    has_body = False
    for p in params:
        loc = p.get("in")
        name = p.get("name")
        param_type = p.get("type")
        if loc in ["path", "query"]:
            cols.append(name)
            param_meta.append({"name": name, "in": loc, "type": param_type})
        elif loc == "formData":
            cols.append(name)
            param_meta.append({
                "name": name, 
                "in": loc, 
                "type": param_type,
                "required": p.get("required", False)
            })
        elif loc == "body":
            has_body = True
    if has_body:
        cols.append("body")
        param_meta.append({"name": "body", "in": "body"})
    # dedupe param_meta by name preserving first occurrence
    seen = set()
    uniq_meta = []
    for m in param_meta:
        if m["name"] not in seen:
            uniq_meta.append(m)
            seen.add(m["name"])
    return cols, uniq_meta

def generate_endpoint_csv(name, columns, output_dir):
    csv_path = os.path.join(output_dir, f"{name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        # Use pipe as delimiter and ensure proper quoting
        writer = csv.writer(f, delimiter='|', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(columns)
        if columns:
            sample_row = []
            for c in columns:
                if c == "host":
                    sample_row.append(DEFAULTS.get("host", "localhost"))
                elif c == "protocol":
                    sample_row.append(DEFAULTS.get("protocol", "https"))
                elif c == "body":
                    sample_row.append("{}")
                else:
                    sample_row.append(f"sample_{c}")
            writer.writerow(sample_row)
    print(f"✅ Test data CSV generated with pipe delimiter: {csv_path}")
    
    # Also update the JMX file to use pipe delimiter
    jmx_path = os.path.join(output_dir, "test.jmx")
    if os.path.exists(jmx_path):
        try:
            with open(jmx_path, 'r', encoding='utf-8') as f:
                content = f.read()
            # Update all CSV DataSet configurations to use pipe delimiter
            updated_content = content.replace('<stringProp name="delimiter">,</stringProp>', 
                                           '<stringProp name="delimiter">|</stringProp>')
            with open(jmx_path, 'w', encoding='utf-8') as f:
                f.write(updated_content)
            print(f"✅ Updated JMX file to use pipe delimiter: {jmx_path}")
        except Exception as e:
            print(f"⚠️ Could not update JMX file: {str(e)}")

# -----------------------------------------------------
# BUILDERS
# -----------------------------------------------------
def build_header_manager(auth_vars):
    if not auth_vars:
        return None

    header_manager = ET.Element("HeaderManager", {
        "guiclass": "HeaderPanel",
        "testclass": "HeaderManager",
        "testname": "Auth Headers",
        "enabled": "true"
    })
    collection = ET.SubElement(header_manager, "collectionProp", {"name": "HeaderManager.headers"})
    for key in auth_vars.keys():
        header = ET.SubElement(collection, "elementProp", {"name": key, "elementType": "Header"})
        ET.SubElement(header, "stringProp", {"name": "Header.name"}).text = key
        ET.SubElement(header, "stringProp", {"name": "Header.value"}).text = f"${{{key}}}"
    return header_manager

def build_httpsampler(endpoint, param_meta, server_config=None):
    path = endpoint["path"]
    method = endpoint["method"].upper()
    # Replace path variables {var} -> ${var}
    path_templated = path
    for m in param_meta:
        if m["in"] == "path":
            path_templated = path_templated.replace("{" + m["name"] + "}", "${" + m["name"] + "}")

    sampler = ET.Element("HTTPSamplerProxy", {
        "guiclass": "HttpTestSampleGui",
        "testclass": "HTTPSamplerProxy",
        "testname": f"{method} {path}",
        "enabled": "true"
    })
    
    # Get server config from endpoint if available
    server_config = endpoint.get('server_config')
    
    # Set default properties that can be overridden by server config
    props = {
        "HTTPSampler.path": path_templated,
        "HTTPSampler.method": method,
    }
    
    # Add domain, protocol, and port from server config if available
    if server_config:
        if 'host' in server_config and server_config['host']:
            props["HTTPSampler.domain"] = server_config['host']
        if 'protocol' in server_config and server_config['protocol']:
            props["HTTPSampler.protocol"] = server_config['protocol']
        if 'port' in server_config and server_config['port']:
            props["HTTPSampler.port"] = str(server_config['port'])
    
    # Set defaults if not provided by server config
    if "HTTPSampler.domain" not in props:
        props["HTTPSampler.domain"] = "${host}"
    if "HTTPSampler.protocol" not in props:
        props["HTTPSampler.protocol"] = "${protocol}"
    for k, v in props.items():
        e = ET.SubElement(sampler, "stringProp", {"name": k})
        e.text = v
    # Proper Arguments container
    args = ET.SubElement(sampler, "elementProp", {"name": "HTTPsampler.Arguments", "elementType": "Arguments"})
    args_coll = ET.SubElement(args, "collectionProp", {"name": "Arguments.arguments"})
    # Handle form data parameters (both regular and file uploads)
    form_data_params = [m for m in param_meta if m.get("in") == "formData"]
    file_uploads = [m for m in form_data_params if m.get("type") == "file"]
    regular_form_data = [m for m in form_data_params if m.get("type") != "file"]
    
    # Add regular form data parameters
    for m in regular_form_data:
        h = ET.SubElement(args_coll, "elementProp", {"name": m["name"], "elementType": "HTTPArgument"})
        ET.SubElement(h, "boolProp", {"name": "HTTPArgument.always_encode"}).text = "false"
        ET.SubElement(h, "stringProp", {"name": "Argument.name"}).text = m["name"]
        ET.SubElement(h, "stringProp", {"name": "Argument.value"}).text = f"${{{m['name']}}}"
        ET.SubElement(h, "stringProp", {"name": "Argument.metadata"}).text = "="
        ET.SubElement(h, "boolProp", {"name": "HTTPArgument.use_equals"}).text = "true"
    
    # Add query parameters
    for m in param_meta:
        if m["in"] == "query":
            h = ET.SubElement(args_coll, "elementProp", {"name": m["name"], "elementType": "HTTPArgument"})
            ET.SubElement(h, "boolProp", {"name": "HTTPArgument.always_encode"}).text = "false"
            ET.SubElement(h, "stringProp", {"name": "Argument.name"}).text = m["name"]
            ET.SubElement(h, "stringProp", {"name": "Argument.value"}).text = f"${{{m['name']}}}"
            ET.SubElement(h, "stringProp", {"name": "Argument.metadata"}).text = "="
            ET.SubElement(h, "boolProp", {"name": "HTTPArgument.use_equals"}).text = "true"
    
    # Common HTTP flags
    ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.follow_redirects"}).text = "true"
    ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.auto_redirects"}).text = "false"
    ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.use_keepalive"}).text = "true"
    
    # Handle file uploads if any
    if file_uploads:
        ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.DO_MULTIPART_POST"}).text = "true"
        
        # Create HTTPFileArgs element for file uploads
        file_args = ET.SubElement(sampler, "elementProp", {"name": "HTTPsampler.FILES", "elementType": "HTTPFileArgs"})
        files_collection = ET.SubElement(file_args, "collectionProp", {"name": "HTTPFileArgs.files"})
        
        for file_param in file_uploads:
            file_elem = ET.SubElement(files_collection, "elementProp", {"name": file_param["name"], "elementType": "HTTPFileArg"})
            ET.SubElement(file_elem, "stringProp", {"name": "File.path"}).text = f"${{{file_param['name']}}}"
            ET.SubElement(file_elem, "stringProp", {"name": "File.paramname"}).text = file_param["name"]
            ET.SubElement(file_elem, "stringProp", {"name": "File.mimetype"}).text = "application/octet-stream"
    else:
        ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.DO_MULTIPART_POST"}).text = "false"
    
    # Handle raw body (if no file uploads)
    has_body = any(m["in"] == "body" for m in param_meta) and method in ["POST", "PUT", "PATCH"]
    ET.SubElement(sampler, "boolProp", {"name": "HTTPSampler.postBodyRaw"}).text = "true" if has_body and not file_uploads else "false"
    
    # Add raw body argument if present and no file uploads
    if has_body and not file_uploads:
        hb = ET.SubElement(args_coll, "elementProp", {"name": "body", "elementType": "HTTPArgument"})
        ET.SubElement(hb, "boolProp", {"name": "HTTPArgument.always_encode"}).text = "false"
        ET.SubElement(hb, "stringProp", {"name": "Argument.name"}).text = ""
        ET.SubElement(hb, "stringProp", {"name": "Argument.value"}).text = "${body_cleaned}"
        ET.SubElement(hb, "stringProp", {"name": "Argument.metadata"}).text = "="
        ET.SubElement(hb, "boolProp", {"name": "HTTPArgument.use_equals"}).text = "true"
    return sampler

def build_throughput_controller(ep_name, sampler, csv_name, columns, method, param_meta):
    ctrl = ET.Element("ThroughputController", {
        "guiclass": "ThroughputControllerGui",
        "testclass": "ThroughputController",
        "testname": ep_name,
        "enabled": "true"
    })
    ET.SubElement(ctrl, "intProp", {"name": "ThroughputController.style"}).text = "1"
    ET.SubElement(ctrl, "stringProp", {"name": "ThroughputController.percentThroughput"}).text = f"${{pct_{ep_name}}}"
    ET.SubElement(ctrl, "boolProp", {"name": "ThroughputController.perThread"}).text = "false"

    hash_tree = ET.Element("hashTree")
    
    # Add sampler first
    hash_tree.append(sampler)
    
    # Create sampler's hashTree for its children
    sampler_tree = ET.Element("hashTree")
    
    # Add CSV DataSet as a child of the sampler
    csv_elem = ET.Element("CSVDataSet", {
        "guiclass": "TestBeanGUI",
        "testclass": "CSVDataSet",
        "testname": f"Data - {ep_name}",
        "enabled": "true"
    })
    ET.SubElement(csv_elem, "stringProp", {"name": "filename"}).text = f"${{testdata.{csv_name}}}"
    ET.SubElement(csv_elem, "stringProp", {"name": "fileEncoding"}).text = "UTF-8"
    ET.SubElement(csv_elem, "stringProp", {"name": "delimiter"}).text = "|"
    ET.SubElement(csv_elem, "boolProp", {"name": "ignoreFirstLine"}).text = "true"
    
    sampler_tree.append(csv_elem)
    sampler_tree.append(ET.Element("hashTree"))  # Required empty hashTree for CSVDataSet

    has_body = any(m["in"] == "body" for m in param_meta) and method in ["POST", "PUT", "PATCH"]
    if has_body:
        """Add a JSR223 PreProcessor for JSON cleaning to the given hash tree"""
        pre_proc = ET.Element("JSR223PreProcessor", {
            "guiclass": "TestBeanGUI",
            "testclass": "JSR223PreProcessor",
            "testname": "Clean JSON Body",
            "enabled": "true"
        })
        ET.SubElement(pre_proc, "stringProp", {"name": "cacheKey"}).text = "true"
        ET.SubElement(pre_proc, "stringProp", {"name": "script"}).text = """// Get the CSV variable
def body = vars.get("body")

if (body != null) {
    // Remove leading and trailing double quotes if present
    body = body.replaceAll(/^"/, "").replaceAll(/"$/, "")

    // Replace all internal double double-quotes with single double-quote
    body = body.replaceAll(/""/, '"')

    // Save back to a new JMeter variable
    vars.put("body_cleaned", body)
}"""
        ET.SubElement(pre_proc, "stringProp", {"name": "scriptLanguage"}).text = "groovy"
        sampler_tree.append(pre_proc)
        sampler_tree.append(ET.Element("hashTree"))  # Empty hashTree for PreProcessor
    
    # Add sampler's children to the main hash tree
    hash_tree.append(sampler_tree)
    
    return ctrl, hash_tree

def generate_jmx(endpoints, auth_vars, output_dir):
    jmx = ET.Element("jmeterTestPlan", {
        "version": "1.2",
        "properties": "5.0",
        "jmeter": "5.6.3"
    })
    # Root hashTree wrapper
    root_tree = ET.SubElement(jmx, "hashTree")
    
    # TestPlan and its hashTree
    tp = ET.SubElement(root_tree, "TestPlan", {
        "guiclass": "TestPlanGui",
        "testclass": "TestPlan",
        "testname": "API Test",
        "enabled": "true"
    })
    ET.SubElement(tp, "stringProp", {"name": "TestPlan.comments"}).text = "Auto-generated Test Plan"
    ET.SubElement(tp, "boolProp", {"name": "TestPlan.functional_mode"}).text = "false"
    ET.SubElement(tp, "boolProp", {"name": "TestPlan.tearDown_on_shutdown"}).text = "true"
    ET.SubElement(tp, "boolProp", {"name": "TestPlan.serialize_threadgroups"}).text = "false"
    ET.SubElement(tp, "stringProp", {"name": "TestPlan.user_define_classpath"}).text = ""
    
    # Required: user defined variables container
    tp_udv = ET.SubElement(tp, "elementProp", {
        "name": "TestPlan.user_defined_variables",
        "elementType": "Arguments",
        "guiclass": "ArgumentsPanel",
        "testclass": "Arguments",
        "testname": "User Defined Variables",
        "enabled": "true"
    })
    ET.SubElement(tp_udv, "collectionProp", {"name": "Arguments.arguments"})

    # HashTree paired with TestPlan
    tp_tree = ET.SubElement(root_tree, "hashTree")

    # Add User Defined Variables
    udv = ET.SubElement(tp_tree, "Arguments", {
        "guiclass": "ArgumentsPanel",
        "testclass": "Arguments",
        "testname": "User Defined Variables",
        "enabled": "true"
    })
    args = ET.SubElement(udv, "collectionProp", name="Arguments.arguments")
    
    # Add default variables
    defaults = {
        "threads": str(DEFAULTS["threads"]),
        "rampup": str(DEFAULTS["rampup"]),
        "duration": str(DEFAULTS["duration"]),
        "loopCount": str(DEFAULTS["loopCount"])
    }
    
    # Add auth variables
    defaults.update(auth_vars)
    
    # Add endpoint percentages
    pct_value = str(round(100 / len(endpoints), 2)) if endpoints else "100.0"
    for ep in endpoints:
        defaults[f"pct_{ep['name']}"] = pct_value
        defaults[f"testdata.{ep['name']}"] = f"${{__P(testdata.{ep['name']}, {ep['name']}.csv)}}"
    
    # Add all variables to the user defined variables
    for name, value in defaults.items():
        arg = ET.SubElement(args, "elementProp", {
            "name": name,
            "elementType": "Argument"
        })
        ET.SubElement(arg, "stringProp", {"name": "Argument.name"}).text = name
        ET.SubElement(arg, "stringProp", {"name": "Argument.value"}).text = str(value)
        ET.SubElement(arg, "stringProp", {"name": "Argument.desc"})
        ET.SubElement(arg, "stringProp", {"name": "Argument.metadata"})
    
    ET.SubElement(tp_tree, "hashTree")  # empty hashTree for User Defined Variables

    thread_group = ET.SubElement(tp_tree, "ThreadGroup", {
        "guiclass": "ThreadGroupGui",
        "testclass": "ThreadGroup",
        "testname": "Test Load Group",
        "enabled": "true"
    })
    # Required main controller for ThreadGroup
    main_ctrl = ET.SubElement(thread_group, "elementProp", {
        "name": "ThreadGroup.main_controller",
        "elementType": "LoopController",
        "guiclass": "LoopControlPanel",
        "testclass": "LoopController",
        "testname": "Loop Controller",
        "enabled": "true"
    })
    ET.SubElement(main_ctrl, "boolProp", {"name": "LoopController.continue_forever"}).text = "false"
    ET.SubElement(main_ctrl, "stringProp", {"name": "LoopController.loops"}).text = "${loopCount}"

    ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.num_threads"}).text = "${threads}"
    ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.ramp_time"}).text = "${rampup}"
    ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.duration"}).text = "${duration}"
    ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.on_sample_error"}).text = "continue"
    ET.SubElement(thread_group, "boolProp", {"name": "ThreadGroup.scheduler"}).text = "true"
    ET.SubElement(thread_group, "stringProp", {"name": "ThreadGroup.delay"}).text = "0"
    
    tg_tree = ET.SubElement(tp_tree, "hashTree")

    # Add HTTP Header Manager for JSON content type and accept headers
    header_manager = ET.SubElement(tg_tree, "HeaderManager", {
        "guiclass": "HeaderPanel",
        "testclass": "HeaderManager",
        "testname": "HTTP Header Manager",
        "enabled": "true"
    })
    headers = ET.SubElement(header_manager, "collectionProp", {"name": "HeaderManager.headers"})
    
    # Add Accept header
    accept_header = ET.SubElement(headers, "elementProp", {
        "name": "",
        "elementType": "Header"
    })
    ET.SubElement(accept_header, "stringProp", {"name": "Header.name"}).text = "Accept"
    ET.SubElement(accept_header, "stringProp", {"name": "Header.value"}).text = "application/json"
    
    # Add Content-Type header
    content_type_header = ET.SubElement(headers, "elementProp", {
        "name": "",
        "elementType": "Header"
    })
    ET.SubElement(content_type_header, "stringProp", {"name": "Header.name"}).text = "Content-Type"
    ET.SubElement(content_type_header, "stringProp", {"name": "Header.value"}).text = "application/json"
    
    # Add empty hashTree for HeaderManager
    ET.SubElement(tg_tree, "hashTree")

    for ep in endpoints:
        csv_name = ep["name"]
        sampler = build_httpsampler(ep, ep.get("param_meta", []))
        ctrl, hash_tree = build_throughput_controller(csv_name, sampler, csv_name, ep.get("columns", []), ep.get("method", "GET"), ep.get("param_meta", []))
        tg_tree.append(ctrl)
        tg_tree.append(hash_tree)

    # Add a single HeaderManager at ThreadGroup level if any auth vars exist
    # header_manager = build_header_manager(auth_vars)
    # if header_manager is not None:
    #     tg_tree.append(header_manager)
    #     tg_tree.append(ET.Element("hashTree"))

    jmx_path = os.path.join(output_dir, "test.jmx")
    with open(jmx_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml(jmx))
    print(f"✅ JMX generated at {jmx_path}")
    return jmx_path

# -----------------------------------------------------
def generate_from_scenario(scenario_data, uploads_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    endpoints = []
    all_auth_vars = {}
    server_configs = {}

    for ep in scenario_data["endpoints"]:
        swagger_path = os.path.join(uploads_dir, ep["file"]) if not os.path.isabs(ep["file"]) else ep["file"]
        swagger = read_json(swagger_path)
        auth_vars = extract_security_definitions(swagger)
        all_auth_vars.update(auth_vars)  # combine from all files

        # Extract server config from swagger's info.x-server-config if it exists
        server_config = None
        if 'info' in swagger and 'x-server-config' in swagger['info']:
            server_config = swagger['info']['x-server-config']
            
            # Store server config by filename for reference
            if ep["file"] not in server_configs:
                server_configs[ep["file"]] = server_config

        name = (
            os.path.splitext(os.path.basename(ep["file"]))[0]
            + ep["path"].split("?")[0].replace("/", "_").replace("{", "").replace("}", "")
            + "_" + ep["method"].lower()
        ).strip("_")

        columns, param_meta = infer_csv_columns(swagger, ep["path"], ep["method"])
        generate_endpoint_csv(name, columns, output_dir)
        
        # Include server config in the endpoint if available
        endpoint_data = {**ep, "name": name, "columns": columns, "param_meta": param_meta}
        if server_config:
            endpoint_data["server_config"] = server_config
            
        endpoints.append(endpoint_data)

    jmx_path = generate_jmx(endpoints, all_auth_vars, output_dir)

    # Collect generated files
    files = [os.path.join(output_dir, f) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))]
    return {"jmx": jmx_path, "files": files, "output_dir": output_dir}