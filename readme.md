# API Test Automation Tool

A comprehensive web-based tool for managing and executing API test scenarios with performance metrics visualization.

## Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Getting Started](#getting-started)
4. [Commands](#commands)

## Overview

This tool provides a user-friendly interface for:
- Managing APIs at once place and prepare test scenarios
- Preparing performance test files for JMeter
- Offline execution of JMeter test files (this tool does not support execution by itself)
- Visualizing test results
- Analyzing API performance metrics

## Features

- **Upload**: 
    1. Upload screen helps adding multiple swagger JSON to the tool (assuming one JSON per microservice). 
    2. Once uploaded base URL of each microservice can be configured.
    3. This page has a button to clear all the test files and test results managed in the tool.
- **Endpoints**: 
    1. Endpoints screen helps listing all the endpoints of all the microservices uploaded in the tool.
    2. Different endpoints in the list can be selected to create a scenario (workflow model).
    3. JMX with a test data CSV can be generated for single endpoint in this page.
    4. User can view the endpoint details in this page.
- **Scenarios**: 
    1. Scenarios screen helps listing all the scenarios created in the tool.
    2. JMX with multiple CSV can be generated for ever scenario in this page.
- **Services**:
    1. This screen will list the microservices uploaded in the tool.
    2. User can generate one JMX with multiple CSVs per microservice in this page.
- **Test Data**:
    1. All the tests generated in this tool (for single endpoint / scenario / microservice) will be listed in this page
    2. Each test will have option to edit the CSVs associated with the test
    3. Limitation: If the test involves uploading file to an endpoint, the test data support this. You can still take the JMX and CSV offline to alter the files.
- **Test Execution**:
    1. Test execution need to be manual. Refer to commands in below sections.
    2. After text execution, you can goto test data page and archive the last run to view the same in Test Results page.
- **Test Results**: 
    1. This page will list all the archived test runs.
    2. User can view the test results in this page.
    3. User can download the test results in this page (to be developed).

## Getting Started

### Prerequisites
- Python 3.8+
- Node.js (for frontend dependencies) # npm install -g api-spec-converter required to handle OpenApi 3.0 specification
- Required Python packages (install via `pip install -r requirements.txt`)

### Installation
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Install node js
4. npm install -g api-spec-converter
5. Run the application: `python app.py`
6. Access the tool at `http://localhost:5000`

## Commands

You can see all the generated tests JMX and the test data CSVs at <<path>>/Test/generated_tests folder

Sample Command

`./jmeter -n -t <<path>>/Test/generated_tests/petstore_get_v2_pet_findByStatus/test.jmx -l <<path>>/Test/generated_tests/petstore_get_v2_pet_findByStatus/results/results.jtl -e -o <<path>>/Test/generated_tests/petstore_get_v2_pet_findByStatus/report`

Running the test to these folders help the tool tool to archive the results appropriately and view in Test Results page.