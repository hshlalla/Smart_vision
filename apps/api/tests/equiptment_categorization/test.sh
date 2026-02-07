#!/bin/bash
# Equipment Classification API Test Script
# Tests multiple equipment classification requests with timing metrics.

# Configuration
API_URL="http://localhost:8000/api/v1/equipment_categorization/predict"
CONTENT_TYPE="application/json"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to make API request and measure time
test_classification() {
    local request_file=$1
    local response_file=$2

    echo "Testing with ${request_file}..."
    
    # Start time measurement
    local start_time=$(date +%s.%N)
    
    if curl --fail -X POST \
        "${API_URL}" \
        -H "Content-Type: ${CONTENT_TYPE}" \
        -d @"${request_file}" \
        -o "${response_file}" \
        --silent; then
        # Calculate execution time
        local end_time=$(date +%s.%N)
        local execution_time=$(echo "$end_time - $start_time" | bc)
        echo -e "${GREEN}Success${NC}: Response saved to ${response_file}"
        echo -e "${BLUE}Time${NC}: ${execution_time} seconds"
    else
        echo -e "${RED}Error${NC}: Failed to process ${request_file}"
        return 1
    fi
}

# Initialize timing variables
total_start_time=$(date +%s.%N)
successful_tests=0
failed_tests=0

# Main test execution
echo "Starting equipment classification tests..."

# Process each test case
for i in {1..5}; do
    request_file="request_$(printf "%02d" $i).json"
    response_file="response_$(printf "%02d" $i).json"
    
    if test_classification "$request_file" "$response_file"; then
        ((successful_tests++))
    else
        ((failed_tests++))
    fi
    
    # Add delay between requests
    if [ $i -lt 5 ]; then
        sleep 1
    fi
done

# Calculate total execution time
total_end_time=$(date +%s.%N)
total_time=$(echo "$total_end_time - $total_start_time" | bc)

# Print summary
echo -e "\nTest Summary:"
echo -e "${GREEN}Successful tests${NC}: $successful_tests"
echo -e "${RED}Failed tests${NC}: $failed_tests"
echo -e "${BLUE}Total execution time${NC}: ${total_time} seconds"