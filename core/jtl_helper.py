import csv
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import gzip

class JtlFileParser:
    """Helper class to parse JTL files and extract timing information."""
    
    # Common JTL column names (JMeter 5.0+ default)
    TIMESTAMP_MS = "timeStamp"
    ELAPSED = "elapsed"
    LABEL = "label"
    RESPONSE_CODE = "responseCode"
    RESPONSE_MESSAGE = "responseMessage"
    THREAD_NAME = "threadName"
    DATA_TYPE = "dataType"
    SUCCESS = "success"
    FAILURE_MESSAGE = "failureMessage"
    BYTES = "bytes"
    SENT_BYTES = "sentBytes"
    GRP_THREADS = "grpThreads"
    ALL_THREADS = "allThreads"
    URL = "URL"
    LATENCY = "Latency"
    IDLE_TIME = "IdleTime"
    CONNECT = "Connect"

    def __init__(self, file_path: str):
        """
        Initialize the JTL file parser.
        
        Args:
            file_path: Path to the JTL file (can be .jtl or .jtl.gz)
        """
        self.file_path = Path(file_path)
        self.column_indices: Dict[str, int] = {}
        self.min_timestamp: Optional[datetime] = None
        self.max_timestamp: Optional[datetime] = None
        self.total_requests: int = 0
        self.successful_requests: int = 0
        self.failed_requests: int = 0

    def _parse_timestamp(self, timestamp_ms: str) -> datetime:
        """Convert timestamp in milliseconds to datetime."""
        return datetime.fromtimestamp(int(timestamp_ms) / 1000.0)

    def _process_row(self, row: Dict[str, str]) -> None:
        """Process a single row of JTL data."""
        if not row or self.TIMESTAMP_MS not in row:
            return

        try:
            timestamp_ms = row[self.TIMESTAMP_MS]
            if not timestamp_ms.strip():
                return

            current_timestamp = self._parse_timestamp(timestamp_ms)
            self.total_requests += 1

            # Update min and max timestamps
            if self.min_timestamp is None or current_timestamp < self.min_timestamp:
                self.min_timestamp = current_timestamp
            if self.max_timestamp is None or current_timestamp > self.max_timestamp:
                self.max_timestamp = current_timestamp

            # Track success/failure
            if self.SUCCESS in row and row[self.SUCCESS].lower() == "true":
                self.successful_requests += 1
            else:
                self.failed_requests += 1

        except (ValueError, TypeError) as e:
            print(f"Warning: Error processing row: {e}")

    def parse(self) -> None:
        """Parse the JTL file to extract timing information."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"JTL file not found: {self.file_path}")

        try:
            # Handle both compressed and uncompressed JTL files
            open_fn = gzip.open if self.file_path.suffix == '.gz' else open
            mode = 'rt' if self.file_path.suffix == '.gz' else 'r'
            
            with open_fn(self.file_path, mode, encoding='utf-8') as f:
                # Read the header to get column indices
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError("Empty JTL file or invalid format")
                
                # Process each row
                for row in reader:
                    self._process_row(row)

        except Exception as e:
            raise Exception(f"Error parsing JTL file: {str(e)}")

    def get_time_range(self) -> Tuple[Optional[datetime], Optional[datetime]]:
        """
        Get the time range of the test.
        
        Returns:
            Tuple of (min_timestamp, max_timestamp)
        """
        return self.min_timestamp, self.max_timestamp

    def get_duration_seconds(self) -> Optional[float]:
        """
        Get the test duration in seconds.
        
        Returns:
            Duration in seconds, or None if timestamps are not available
        """
        if self.min_timestamp and self.max_timestamp:
            return (self.max_timestamp - self.min_timestamp).total_seconds()
        return None

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the JTL file.
        
        Returns:
            Dictionary containing summary information
        """
        return {
            "file_path": str(self.file_path),
            "min_timestamp": self.min_timestamp.isoformat() if self.min_timestamp else None,
            "max_timestamp": self.max_timestamp.isoformat() if self.max_timestamp else None,
            "duration_seconds": self.get_duration_seconds(),
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (self.successful_requests / self.total_requests * 100) if self.total_requests > 0 else 0
        }

# Example usage
if __name__ == "__main__":
    # Example usage
    jtl_parser = JtlFileParser("/Users/rakeshvijayakumar/Documents/Business/SmartSolutions/Test/db/tests/combo/results_20251027_115351.jtl")
    try:
        jtl_parser.parse()
        min_ts, max_ts = jtl_parser.get_time_range()
        duration = jtl_parser.get_duration_seconds()
        
        print(f"Test started at: {min_ts}")
        print(f"Test ended at: {max_ts}")
        print(f"Test duration: {duration:.2f} seconds")
        
        summary = jtl_parser.get_summary()
        print("\nTest Summary:")
        for key, value in summary.items():
            print(f"{key}: {value}")
            
    except Exception as e:
        print(f"Error: {str(e)}")