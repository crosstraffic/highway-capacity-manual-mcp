<h1 align="center">HCM-LLM MCP Server</h1>

A FastAPI-based Model Context Protocol (MCP) server for Highway Capacity Manual (HCM) analysis and transportation engineering calculations. So far, this server provides comprehensive two-lane highway analysis following HCM Chapter 15 methodology.

## Features

- Complete HCM Chapter 15 two-lane highway analysis
- YAML-based function registry for easy extensibility
- Function calling interface with 15+ transportation analysis functions
- Semantic search over HCM documentation using ChromaDB
- MCP server compatibility for integration with AI assistants
- RESTful API endpoints for direct access
- Dynamic endpoint generation based on registry
- Comprehensive test suite and validation tools

## Installation
```bash
# Windows
.venv\Scripts\activate

pip install fastapi uvicorn sentence-transformers chromadb fastapi-mcp transportations-library pyyaml python-dotenv

# OR
install from requirements.txt:

# Then
pip install -r requirements.txt
```

## Quick Start

```bash
# 1. Setup the database:
python scripts/setup_database.py

# 2. Validate the registry:
python scripts/validate_registry.py

# 3. Start the server:
python main.py
```

4. Access the API at `http://localhost:8000`
5. View interactive documentation at `http://localhost:8000/docs`

## Project Structure

```
hcm-llm-api/
├── main.py                     # Main FastAPI application
├── functions_registry.yaml     # Function registry configuration
├── src/
│   ├── core/                       # Core application modules
│   │   ├── registry.py             # Function registry implementation
│   │   ├── models.py               # Pydantic data models
│   │   └── endpoints.py            # Dynamic endpoint creation
│   ├── functions/                  # Function implementations by chapter
│   │   ├── chapter15.py            # Chapter 15: Two-Lane Highways
│   │   └── research.py             # Research and documentation
│   └── scripts/                    # Utility scripts
│       ├── setup_database.py       # ChromaDB setup
│       └── validate_registry.py    # Registry validation
└── data/                           # Registry validation
    ├── hcm_documents/              # HCM documentation files
    └── chroma_db/                  # ChromaDB storage  

```

## Configuration

### Environment Variables
Create a `.env` file based on `.env.example`:
```
CHROMA_DB_PATH=./chroma_db
HOST=127.0.0.1
PORT=8000
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
LOG_LEVEL=INFO
```

### Function Registry
Functions are defined in `functions_registry.yaml`:

```yaml
functions:
  chapter15:
    identify_vertical_class:
      module: "functions.chapter15"
      function: "identify_vertical_class_function"
      description: "Identify vertical alignment class range"
      category: "transportation"
      chapter: 15
      step: 1
      parameters:
        type: "object"
        properties:
          segment_index:
            type: "integer"
          highway_data:
            type: "object"
        required: ["segment_index", "highway_data"]
```

## API Usage

### Complete Highway Analysis

```bash
curl -X POST "http://localhost:8000/analysis/chapter15/complete" \
  -H "Content-Type: application/json" \
  -d '{
    "segments": [{
      "passing_type": 0,
      "length": 2.0,
      "grade": 2.0,
      "spl": 50.0,
      "volume": 760.0,
      "volume_op": 1500.0,
      "phf": 0.95,
      "phv": 5.0
    }],
    "lane_width": 12.0,
    "shoulder_width": 6.0,
    "apd": 5.0
  }'
```

### Function Calling Interface

```bash
curl -X POST "http://localhost:8000/tools/call" \
  -H "Content-Type: application/json" \
  -d '{
    "function": {
      "name": "chapter15_determine_free_flow_speed",
      "arguments": {
        "segment_index": 0,
        "highway_data": {
          "segments": [{"passing_type": 0, "length": 2.0, "grade": 2.0, "spl": 50.0}],
          "lane_width": 12.0,
          "shoulder_width": 6.0
        }
      }
    }
  }'
```

### List Available Functions

```bash
# List all functions
curl -X POST "http://localhost:8000/tools/list"

# Filter by category
curl -X POST "http://localhost:8000/tools/list" \
  -H "Content-Type: application/json" \
  -d '{"category": "transportation"}'

# Filter by chapter
curl -X POST "http://localhost:8000/tools/list" \
  -H "Content-Type: application/json" \
  -d '{"chapter": 15}'
```

### Query HCM Documentation

```bash
curl -X POST "http://localhost:8000/tools/query-hcm" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What factors affect free flow speed in two-lane highways?",
    "top_k": 5
  }'
```

## Available Functions

### Chapter 15 Functions
- `chapter15_identify_vertical_class` - Step 1: Identify vertical alignment class range
- `chapter15_determine_demand_flow` - Step 2: Calculate demand flow rates and capacity
- `chapter15_determine_vertical_alignment` - Step 3: Determine vertical alignment classification
- `chapter15_determine_free_flow_speed` - Step 4: Calculate free flow speed
- `chapter15_estimate_average_speed` - Step 5: Estimate average travel speed
- `chapter15_estimate_percent_followers` - Step 6: Estimate percentage of following vehicles
- `chapter15_determine_follower_density_pl` - Step 8a: Follower density for passing lanes
- `chapter15_determine_follower_density_pc_pz` - Step 8b: Follower density for PC/PZ segments
- `chapter15_determine_segment_los` - Step 9: Calculate segment Level of Service
- `chapter15_determine_facility_los` - Step 10: Calculate facility Level of Service
- `chapter15_complete_analysis` - Complete HCM Chapter 15 procedure

### Research Functions
- `query_hcm` - Query HCM documentation database
- `search_hcm_by_chapter` - Search HCM content by specific chapter
- `get_hcm_section` - Get specific HCM section content
- `summarize_hcm_content` - Summarize HCM content for a topic

## MCP Server Usage

### Claude Desktop Integration

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "hcm-analysis": {
      "command": "python",
      "args": ["path/to/main.py"],
      "env": {
        "PORT": "8000",
        "CHROMA_DB_PATH": "./chroma_db"
      }
    }
  }
}
```

### Custom AI Assistant Integration

The server exposes MCP discovery at `/mcp/discovery` and implements standard MCP protocols for:
- Tool discovery and execution via function registry
- Resource access (HCM documentation)
- Dynamic capability reporting

## API Endpoints

### Core Endpoints
- `POST /tools/call` - Execute any registered function
- `POST /tools/list` - List available functions with filtering
- `GET /mcp/discovery` - MCP capability discovery

### Chapter 15 Analysis
- `POST /analysis/chapter15/complete` - Complete HCM analysis
- `POST /analysis/chapter15/segment` - Single segment analysis

### Research
- `POST /tools/query-hcm` - Query HCM database

### Utility
- `GET /health` - Health check
- `GET /registry/info` - Registry information
- `POST /registry/reload` - Reload function registry

## Data Models

### Highway Segment
```python
{
  "passing_type": 0,      # 0=PC, 1=PZ, 2=PL
  "length": 2.0,          # miles
  "grade": 2.0,           # percent
  "spl": 50.0,            # speed limit (mph)
  "volume": 760.0,        # vehicles/hour
  "volume_op": 1500.0,    # opposing volume
  "phf": 0.95,            # peak hour factor
  "phv": 5.0              # percent heavy vehicles
}
```

### Highway Facility
```python
{
  "segments": [...],      # list of segments
  "lane_width": 12.0,     # feet
  "shoulder_width": 6.0,  # feet
  "apd": 5.0,             # access points/mile
  "pmhvfl": 0.02,         # percent HV in fast lane
  "l_de": 0.0             # effective passing distance
}
```

## Adding New HCM Chapters

### 1. Create Function Module
Create `functions/chapter16.py`:

```python
def new_analysis_function(data: Dict[str, Any]) -> Dict[str, Any]:
    """Implementation for new analysis."""
    try:
        # Your implementation here
        return {"success": True, "result": "analysis_result"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

### 2. Update Registry
Add to `functions_registry.yaml`:

```yaml
functions:
  chapter16:
    new_analysis:
      module: "functions.chapter16"
      function: "new_analysis_function"
      description: "New analysis function"
      category: "transportation"
      chapter: 16
      parameters:
        type: "object"
        properties:
          input_param:
            type: "string"
        required: ["input_param"]
```

### 3. Restart Server
The registry will automatically load the new functions.

## Development

### Running Tests
```bash
pytest tests/
```

### Validating Registry
```bash
python scripts/validate_registry.py
```

### Setting Up Development Database
```bash
python scripts/setup_database.py
```

## Customization

### Custom Analysis Models
Extend models in `core/models.py`:

```python
class CustomAnalysisInput(BaseModel):
    parameter1: float = Field(description="Custom parameter")
    parameter2: str = Field(description="Another parameter")
```

### Custom Functions
1. Implement function in appropriate module
2. Add to `functions_registry.yaml`
3. Restart server or call `/registry/reload`

### Database Integration
To add HCM content:
1. Place documents in `data/hcm_documents/`
2. Run `python scripts/import_hcm_docs.py`
3. Query using `/tools/query-hcm`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add functions to appropriate chapter module
4. Update `functions_registry.yaml`
5. Add tests in `tests/`
6. Run validation: `python scripts/validate_registry.py`
7. Ensure all tests pass: `pytest`
8. Submit a pull request

## Support

For issues and questions:
- Open an issue on GitHub
- Check the API documentation at `/docs`
- Review function registry at `/registry/info`
- Validate setup with utility scripts

2. Access the API at `http://localhost:8000`
3. View interactive documentation at `http://localhost:8000/docs`