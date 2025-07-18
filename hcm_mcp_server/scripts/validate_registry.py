import yaml
import importlib
from pathlib import Path


def validate_registry(registry_file: Path = Path("functions_registry.yaml")):
    """Validate the function registry configuration."""
    print(f"Validating registry: {registry_file}")
    
    if not registry_file.exists():
        print(f"ERROR: Registry file not found: {registry_file}")
        return False
    
    try:
        with open(registry_file, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"ERROR: Invalid YAML: {e}")
        return False
    
    functions_config = config.get('functions', {})
    errors = []
    warnings = []
    
    for chapter, chapter_functions in functions_config.items():
        print(f"\nValidating chapter: {chapter}")
        
        for func_name, func_config in chapter_functions.items():
            print(f"  Checking function: {func_name}")
            
            # Check required fields
            required_fields = ['module', 'function', 'description', 'parameters']
            for field in required_fields:
                if field not in func_config:
                    errors.append(f"{chapter}.{func_name}: Missing required field '{field}'")
            
            # Check module exists and function is available
            module_path = func_config.get('module')
            function_name = func_config.get('function')
            
            if module_path and function_name:
                try:
                    module = importlib.import_module(module_path)
                    if not hasattr(module, function_name):
                        errors.append(f"{chapter}.{func_name}: Function '{function_name}' not found in module '{module_path}'")
                    else:
                        print("    ✓ Module and function found")
                except ImportError as e:
                    warnings.append(f"{chapter}.{func_name}: Could not import module '{module_path}': {e}")
            
            # Validate parameters schema
            parameters = func_config.get('parameters', {})
            if not isinstance(parameters, dict):
                errors.append(f"{chapter}.{func_name}: Parameters must be a dictionary")
            elif 'type' not in parameters:
                warnings.append(f"{chapter}.{func_name}: Parameters should have 'type' field")
    
    # Print results
    print("\nValidation Results:")
    print(f"Errors: {len(errors)}")
    print(f"Warnings: {len(warnings)}")
    
    if errors:
        print("\nERRORS:")
        for error in errors:
            print(f"  - {error}")
    
    if warnings:
        print("\nWARNINGS:")
        for warning in warnings:
            print(f"  - {warning}")
    
    return len(errors) == 0


def main():
    """Main validation function."""
    registry_file = Path("functions_registry.yaml")
    
    if validate_registry(registry_file):
        print("\n✓ Registry validation passed!")
        return 0
    else:
        print("\n✗ Registry validation failed!")
        return 1


if __name__ == "__main__":
    exit(main())