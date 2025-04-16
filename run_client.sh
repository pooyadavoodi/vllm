set -x 

curl -s http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
  "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
  "prompt": "San Francisco is a",
  "max_tokens": 1000
}'