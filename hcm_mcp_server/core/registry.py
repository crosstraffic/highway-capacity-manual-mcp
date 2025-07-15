import yaml
import importlib
from typing import Dict, Any, Callable, Optional, List
from pathlib import Path


class FunctionRegistry:
    """Manages function registration and discovery from YAML configuration."""
    
    def __init__(self, registry_file: Path):
        self.registry_file = registry_file
        self.functions: Dict[str, Dict[str, Any]] = {}
        self.modules: Dict[str, Any] = {}
        self.load_registry()
    
    def load_registry(self) -> None:
        """Load function registry from YAML file."""
        if not self.registry_file.exists():
            print(f"Warning: Registry file not found: {self.registry_file}")
            print("Creating empty registry...")

            self.config = {}
            self.categories = {}
            self.functions = {}
            return
            # raise FileNotFoundError(f"Registry file not found: {self.registry_file}")
        
        try:
            with open(self.registry_file, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading registry file: {e}")
            self.config = {}
            self.categories = {}
            self.functions = {}
            return
        
        self.config = config.get('config', {})
        self.categories = config.get('categories', {})
        
        # Load all functions from all chapters
        functions_config = config.get('functions', {})
        
        for chapter, chapter_functions in functions_config.items():
            if not isinstance(chapter_functions, dict):
                continue

            for func_name, func_config in chapter_functions.items():
                full_name = f"{chapter}_{func_name}" if chapter != 'research' else func_name
                self.register_function(full_name, func_config)
    
    def register_function(self, name: str, config: Dict[str, Any]) -> None:
        """Register a single function from configuration."""
        if not isinstance(config, dict):
            print(f"Warning: Invalid config for function {name}: {config}")
            return

        module_path = config['module']
        function_name = config['function']
        
        # Load module if not already loaded
        if module_path not in self.modules:
            try:
                self.modules[module_path] = importlib.import_module(module_path)
            except ImportError as e:
                print(f"Warning: Could not import module {module_path}: {e}")
                # Store a placeholder function
                self.functions[name] = {
                    'function': lambda data: {"success": False, "error": f"Function {function_name} not found"},
                    'description': config.get('description', 'Function not available'),
                    'category': config.get('category', 'general'),
                    'chapter': config.get('chapter'),
                    'step': config.get('step'),
                    'parameters': config.get('parameters', {}),
                    'module': module_path,
                    'comprehensive': config.get('comprehensive', False),
                    'available': False
                }
                return
        
        # Get function from module
        module = self.modules[module_path]
        if not hasattr(module, function_name):
            print(f"Warning: Function {function_name} not found in module {module_path}")
            self.functions[name] = {
                'function': lambda data: {"success": False, "error": f"Function {function_name} not found"},
                'description': config.get('description', 'Function not available'),
                'category': config.get('category', 'general'),
                'chapter': config.get('chapter'),
                'step': config.get('step'),
                'parameters': config.get('parameters', {}),
                'module': module_path,
                'comprehensive': config.get('comprehensive', False),
                'available': False
            }
            return
        
        function_impl = getattr(module, function_name)
        
        # Store complete function information
        self.functions[name] = {
            'function': function_impl,
            'description': config['description'],
            'category': config.get('category', 'general'),
            'chapter': config.get('chapter'),
            'step': config.get('step'),
            'parameters': config['parameters'],
            'module': module_path,
            'comprehensive': config.get('comprehensive', False)
        }
    
    def get_function(self, name: str) -> Optional[Callable]:
        """Get a registered function by name."""
        if name in self.functions:
            return self.functions[name]['function']
        return None
    
    def get_function_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get complete information about a function."""
        return self.functions.get(name)
    
    def get_all_functions(self) -> Dict[str, Dict[str, Any]]:
        """Get all registered functions."""
        return self.functions
    
    def get_functions_by_category(self, category: str) -> Dict[str, Dict[str, Any]]:
        """Get functions filtered by category."""
        return {
            name: info for name, info in self.functions.items()
            if info.get('category') == category
        }
    
    def get_functions_by_chapter(self, chapter: int) -> Dict[str, Dict[str, Any]]:
        """Get functions filtered by HCM chapter."""
        return {
            name: info for name, info in self.functions.items()
            if info.get('chapter') == chapter
        }
    
    def list_categories(self) -> List[str]:
        """List all available categories."""
        categories = set()
        for func_info in self.functions.values():
            categories.add(func_info.get('category', 'general'))
        return sorted(list(categories))
    
    def list_chapters(self) -> List[int]:
        """List all available HCM chapters."""
        chapters = set()
        for func_info in self.functions.values():
            if func_info.get('chapter'):
                chapters.add(func_info['chapter'])
        return sorted(list(chapters))
    
    def validate_function_parameters(self, name: str, parameters: Dict[str, Any]) -> bool:
        """Validate function parameters against schema."""
        func_info = self.get_function_info(name)
        if not func_info:
            return False
        
        param_schema = func_info.get('parameters', {})
        required_params = param_schema.get('required', [])
        
        # Check required parameters
        for required in required_params:
            if required not in parameters:
                return False
        
        return True
    
    def get_function_dependencies(self, name: str) -> List[str]:
        """Get function dependencies based on HCM steps."""
        func_info = self.get_function_info(name)
        if not func_info:
            return []
        
        chapter = func_info.get('chapter')
        step = func_info.get('step')
        
        if not chapter or not step:
            return []
        
        # Find all functions from same chapter with lower step numbers
        dependencies = []
        for func_name, info in self.functions.items():
            if (info.get('chapter') == chapter and 
                info.get('step') and 
                info['step'] < step):
                dependencies.append(func_name)
        
        return sorted(dependencies, key=lambda x: self.functions[x].get('step', 0))
    
    def reload_registry(self) -> None:
        """Reload the registry from file."""
        self.functions.clear()
        self.modules.clear()
        self.load_registry()