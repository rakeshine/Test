import boto3
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
from dataclasses import dataclass

@dataclass
class MetricDataPoint:
    timestamp: datetime
    value: float
    unit: str

class CloudWatchMetricsFetcher:
    def __init__(self, region_name: str = 'us-east-1'):
        """
        Initialize the CloudWatch client.
        
        Args:
            region_name: AWS region name (default: 'us-east-1')
        """
        self.client = boto3.client('cloudwatch', region_name=region_name)
        self.ecs_client = boto3.client('ecs', region_name=region_name)
    
    def get_ecs_service_metrics(
        self,
        cluster_name: str,
        service_name: str,
        start_time: datetime,
        end_time: datetime,
        period: int = 300,
        statistics: List[str] = ['Average', 'Maximum']
    ) -> Dict[str, List[MetricDataPoint]]:
        """
        Get CPU and Memory utilization metrics for an ECS service.
        
        Args:
            cluster_name: Name of the ECS cluster
            service_name: Name of the ECS service
            start_time: Start time for the metrics
            end_time: End time for the metrics
            period: The granularity, in seconds, of the returned data points
            statistics: List of statistics to include (e.g., ['Average', 'Maximum'])
            
        Returns:
            Dictionary containing CPU and Memory metrics
        """
        # Get the ECS service to find the service namespace
        try:
            service = self.ecs_client.describe_services(
                cluster=cluster_name,
                services=[service_name]
            )['services'][0]
            
            if not service:
                raise ValueError(f"Service {service_name} not found in cluster {cluster_name}")
                
        except Exception as e:
            raise Exception(f"Error fetching ECS service details: {str(e)}")
        
        # Get the service namespace from the service ARN
        service_namespace = service['serviceArn'].split(':')[4]
        
        # Define the metric queries
        metric_queries = [
            {
                'Id': 'cpu_utilization',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/ECS',
                        'MetricName': 'CPUUtilization',
                        'Dimensions': [
                            {
                                'Name': 'ClusterName',
                                'Value': cluster_name
                            },
                            {
                                'Name': 'ServiceName',
                                'Value': service_name
                            }
                        ]
                    },
                    'Period': period,
                    'Stat': 'Average',
                    'Unit': 'Percent'
                },
                'ReturnData': True
            },
            {
                'Id': 'memory_utilization',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'AWS/ECS/ContainerInsights',
                        'MetricName': 'MemoryUtilized',
                        'Dimensions': [
                            {
                                'Name': 'ClusterName',
                                'Value': cluster_name
                            },
                            {
                                'Name': 'ServiceName',
                                'Value': service_name
                            }
                        ]
                    },
                    'Period': period,
                    'Stat': 'Average',
                    'Unit': 'Bytes'
                },
                'ReturnData': True
            }
        ]
        
        # Get the metric data
        try:
            response = self.client.get_metric_data(
                MetricDataQueries=metric_queries,
                StartTime=start_time,
                EndTime=end_time,
                ScanBy='TimestampAscending'
            )
            
            # Process the results
            results = {}
            for result in response['MetricDataResults']:
                metric_name = result['Label']
                timestamps = result['Timestamps']
                values = result['Values']
                unit = result.get('Unit', 'None')
                
                data_points = [
                    MetricDataPoint(timestamp=ts, value=val, unit=unit)
                    for ts, val in zip(timestamps, values)
                ]
                
                results[metric_name] = data_points
                
            return results
            
        except Exception as e:
            raise Exception(f"Error fetching CloudWatch metrics: {str(e)}")
    
    def get_ecs_container_metrics(
        self,
        cluster_name: str,
        service_name: str,
        container_name: str,
        start_time: datetime,
        end_time: datetime,
        period: int = 300
    ) -> Dict[str, List[MetricDataPoint]]:
        """
        Get CPU and Memory utilization metrics for a specific container in an ECS service.
        
        Args:
            cluster_name: Name of the ECS cluster
            service_name: Name of the ECS service
            container_name: Name of the container
            start_time: Start time for the metrics
            end_time: End time for the metrics
            period: The granularity, in seconds, of the returned data points
            
        Returns:
            Dictionary containing CPU and Memory metrics
        """
        # Define the metric queries
        metric_queries = [
            {
                'Id': 'cpu_utilization',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'ECS/ContainerInsights',
                        'MetricName': 'CpuUtilized',
                        'Dimensions': [
                            {
                                'Name': 'ClusterName',
                                'Value': cluster_name
                            },
                            {
                                'Name': 'ServiceName',
                                'Value': service_name
                            },
                            {
                                'Name': 'ContainerName',
                                'Value': container_name
                            }
                        ]
                    },
                    'Period': period,
                    'Stat': 'Average',
                    'Unit': 'Percent'
                },
                'ReturnData': True
            },
            {
                'Id': 'memory_utilization',
                'MetricStat': {
                    'Metric': {
                        'Namespace': 'ECS/ContainerInsights',
                        'MetricName': 'MemoryUtilized',
                        'Dimensions': [
                            {
                                'Name': 'ClusterName',
                                'Value': cluster_name
                            },
                            {
                                'Name': 'ServiceName',
                                'Value': service_name
                            },
                            {
                                'Name': 'ContainerName',
                                'Value': container_name
                            }
                        ]
                    },
                    'Period': period,
                    'Stat': 'Average',
                    'Unit': 'Bytes'
                },
                'ReturnData': True
            }
        ]
        
        # Get the metric data
        try:
            response = self.client.get_metric_data(
                MetricDataQueries=metric_queries,
                StartTime=start_time,
                EndTime=end_time,
                ScanBy='TimestampAscending'
            )
            
            # Process the results
            results = {}
            for result in response['MetricDataResults']:
                metric_name = result['Label']
                timestamps = result['Timestamps']
                values = result['Values']
                unit = result.get('Unit', 'None')
                
                data_points = [
                    MetricDataPoint(timestamp=ts, value=val, unit=unit)
                    for ts, val in zip(timestamps, values)
                ]
                
                results[metric_name] = data_points
                
            return results
            
        except Exception as e:
            raise Exception(f"Error fetching container metrics: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize the fetcher
    fetcher = CloudWatchMetricsFetcher(region_name='your-region')
    
    # Set the time range (last 1 hour)
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(hours=1)
    
    try:
        # Get service-level metrics
        print("Fetching service metrics...")
        service_metrics = fetcher.get_ecs_service_metrics(
            cluster_name='your-cluster-name',
            service_name='your-service-name',
            start_time=start_time,
            end_time=end_time
        )
        
        # Print service metrics
        for metric_name, data_points in service_metrics.items():
            print(f"\n{metric_name}:")
            for dp in data_points:
                print(f"  {dp.timestamp}: {dp.value} {dp.unit}")
        
        # Get container-level metrics
        print("\nFetching container metrics...")
        container_metrics = fetcher.get_ecs_container_metrics(
            cluster_name='your-cluster-name',
            service_name='your-service-name',
            container_name='your-container-name',
            start_time=start_time,
            end_time=end_time
        )
        
        # Print container metrics
        for metric_name, data_points in container_metrics.items():
            print(f"\n{metric_name}:")
            for dp in data_points:
                print(f"  {dp.timestamp}: {dp.value} {dp.unit}")
                
    except Exception as e:
        print(f"Error: {str(e)}")