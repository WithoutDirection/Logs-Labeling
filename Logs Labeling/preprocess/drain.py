import re
import pandas as pd
from typing import List, Tuple, Optional, Dict


class LogCluster:
    """Represents a cluster of similar log messages sharing a template."""
    
    def __init__(self, template: List[str]):
        self.template = template
        self.count = 1
    
    def __repr__(self):
        return f"LogCluster('{' '.join(self.template)}', count={self.count})"


class Node:
    """Node in the Drain parse tree."""
    
    def __init__(self):
        self.children = {}  # Dict[str, Node] or List[LogCluster] at leaf level


class DrainParser:
    """
    Drain log parser for extracting templates and parameters.
    Supports both standard logs and Windows Registry operations.
    
    Parameters:
        depth: Parse tree depth (default: 4, excludes root and length layers)
        st: Similarity threshold for cluster matching (0-1, default: 0.5)
        max_children: Max children per node (default: 100)
        rex: Regex patterns for preprocessing
        registry_mode: Enable registry-specific parsing (default: False)
    """
    
    # Registry operation categories
    QUERY_OPS = {'RegQueryValue', 'RegQueryKey', 'RegEnumKey', 'RegEnumValue', 'RegOpenKey'}
    WRITE_OPS = {'RegSetValue', 'RegCreateKey', 'RegSetInfoKey'}
    DELETE_OPS = {'RegDeleteKey', 'RegDeleteValue'}
    CLOSE_OPS = {'RegCloseKey'}
    
    def __init__(
        self,
        depth: int = 4,
        st: float = 0.5,
        max_children: int = 100,
        rex: Optional[List[str]] = None,
        registry_mode: bool = False
    ):
        self.depth = depth - 2  # Exclude root and length layers
        self.st = st
        self.max_children = max_children
        self.registry_mode = registry_mode
        
        # Patterns with descriptive tags: (pattern, tag)
        # Order matters: more specific patterns first
        self.rex_patterns = rex or [
            (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', '<IP>'),
            (r'\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b', '<UUID>'),
            (r'\b0x[0-9a-fA-F]+\b', '<HEX>'),
            (r'\bhttps?://[^\s\'"\\]+', '<URL>'),  # URLs (before paths)
            (r'[A-Z]:\\[\w\\.-]+', '<PATH>'),  # Windows path
            (r'(?<![:/])/(?:[/\w.-]+/)*[\w.-]+', '<PATH>'),  # Unix path
            (r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}', '<TIMESTAMP>'),
            (r'\b\d+\.\d+\b', '<FLOAT>'),
            (r'\b\d+\b', '<NUM>'),
        ]
        
        self.root = Node()
        self.clusters = []
    
    @staticmethod
    def is_wildcard(token: str) -> bool:
        """Check if token is a wildcard (excludes functional tags like <ROOT:HKLM>)."""
        return token.startswith('<') and token.endswith('>') and ':' not in token
    
    @staticmethod
    def is_registry_operation(operation: str) -> bool:
        """Check if operation is a registry operation."""
        if not operation:
            return False
        return operation.startswith('Reg')
    
    def parse_log_row(self, row: pd.Series, columns: List[str]) -> Tuple[str, List[str], str]:
        """Parse a single log row.
        
        Args:
            row: DataFrame row containing log data
            columns: List of column names to parse
            
        Returns:
            (template, parameters, original_log_message)
        """
        # Build log message from columns
        log_parts = []
        for col in columns:
            if col in row.index:
                val = row[col]
                if pd.notna(val) and str(val).strip() and str(val).lower() != 'nan':
                    log_parts.append(str(val).strip())
        log_message = " ".join(log_parts)
        
        # Parse using appropriate method
        if self.registry_mode:
            # Use structured parsing for registry events
            operation = row.get('Operation', '')
            if self.is_registry_operation(operation):
                template, params = self.parse_from_row(row.to_dict())
            else:
                template, params = self.parse(log_message)
        else:
            # Standard parsing
            template, params = self.parse(log_message)
        
        return template, params, log_message
    
    def preprocess(self, message: str) -> str:
        """Apply regex substitutions to normalize variable content."""
        for pattern, tag in self.rex_patterns:
            message = re.sub(pattern, tag, message)
        return message
    
    def tokenize(self, message: str) -> List[str]:
        """Split message into tokens."""
        return message.strip().split()
    
    def search_tree(self, tokens: List[str]) -> List[LogCluster]:
        """Search parse tree for candidate clusters."""
        length = len(tokens)
        
        # Layer 1: Group by length
        if length not in self.root.children:
            return []
        
        node = self.root.children[length]
        
        # Layer 2 to depth: Navigate by tokens
        for idx, token in enumerate(tokens):
            if idx >= self.depth:
                break
            
            # Try exact match first
            if token in node.children:
                node = node.children[token]
            else:
                # Try wildcard match (any <TAG>)
                wildcard_found = False
                for key in node.children:
                    if self.is_wildcard(key):
                        node = node.children[key]
                        wildcard_found = True
                        break
                if not wildcard_found:
                    return []
        
        # Return leaf clusters
        return node.children.get('clusters', [])
    
    def add_cluster_to_tree(self, cluster: LogCluster):
        """Add new cluster to parse tree."""
        tokens = cluster.template
        length = len(tokens)
        
        # Layer 1: Group by length
        if length not in self.root.children:
            self.root.children[length] = Node()
        
        node = self.root.children[length]
        
        # Layer 2 to depth: Build path
        for idx, token in enumerate(tokens):
            if idx >= self.depth:
                break
            
            # Use the token as key directly (wildcards stay as wildcards, text stays as text)
            key = token
            
            if key not in node.children:
                node.children[key] = Node()
            node = node.children[key]
        
        # Leaf level: Store cluster
        if 'clusters' not in node.children:
            node.children['clusters'] = []
        node.children['clusters'].append(cluster)
    
    def calculate_similarity(self, template: List[str], tokens: List[str]) -> Tuple[float, int]:
        """
        Calculate similarity between template and tokens.
        Returns (similarity_ratio, wildcard_count).
        """
        if len(template) != len(tokens):
            return 0.0, 0
        
        matches = 0
        wildcards = 0
        
        for t, s in zip(template, tokens):
            if self.is_wildcard(t):
                wildcards += 1
                # Wildcard matches if token is same type or both are wildcards
                if t == s or self.is_wildcard(s):
                    matches += 1
            elif t == s:
                matches += 1
        
        return matches / len(template), wildcards
    
    def find_best_cluster(self, clusters: List[LogCluster], tokens: List[str]) -> Optional[LogCluster]:
        """Find best matching cluster above similarity threshold."""
        best_cluster = None
        best_sim = -1
        best_wildcards = -1
        
        for cluster in clusters:
            sim, wildcards = self.calculate_similarity(cluster.template, tokens)
            
            # Prefer higher similarity, then more wildcards (more general)
            if sim > best_sim or (sim == best_sim and wildcards > best_wildcards):
                best_sim = sim
                best_wildcards = wildcards
                best_cluster = cluster
        
        return best_cluster if best_sim >= self.st else None
    
    def merge_templates(self, template1: List[str], template2: List[str]) -> List[str]:
        """Merge two templates, preserving wildcard types when possible."""
        result = []
        for t1, t2 in zip(template1, template2):
            if t1 == t2:
                result.append(t1)
            elif self.is_wildcard(t1):
                result.append(t1)  # Keep existing wildcard type
            elif self.is_wildcard(t2):
                result.append(t2)  # Use new wildcard type
            else:
                # Keep the original text from template1 when they differ
                result.append(t1)
        return result
    
    def extract_parameters(self, template: List[str], original_tokens: List[str]) -> List[str]:
        """Extract parameters corresponding to wildcards in template."""
        parameters = []
        
        for temp_token, orig_token in zip(template, original_tokens):
            if self.is_wildcard(temp_token):
                parameters.append(orig_token)
            elif orig_token != temp_token:
                # Extract embedded wildcards (e.g., DownloadFile(\"<URL>\",)
                wildcards = re.findall(r'<([A-Z]+)>', temp_token)
                for wildcard_name in wildcards:
                    if ':' not in f'<{wildcard_name}>':  # Skip functional tags
                        for pattern, tag in self.rex_patterns:
                            if tag == f'<{wildcard_name}>':
                                parameters.extend(re.findall(pattern, orig_token))
                                break
        
        return parameters
    
    def parse(self, log_message: str) -> Tuple[str, List[str]]:
        """Parse log message into template and parameters."""
        original_tokens = self.tokenize(log_message)
        processed = self.preprocess(log_message)
        tokens = self.tokenize(processed)
        
        if not tokens:
            return "", []
        
        # Search for existing cluster
        candidates = self.search_tree(tokens)
        cluster = self.find_best_cluster(candidates, tokens)
        
        if cluster:
            cluster.count += 1
            cluster.template = self.merge_templates(cluster.template, tokens)
        else:
            cluster = LogCluster(tokens.copy())
            self.clusters.append(cluster)
            self.add_cluster_to_tree(cluster)
        
        template = ' '.join(cluster.template)
        parameters = self.extract_parameters(cluster.template, original_tokens)
        
        return template, parameters
    
    def get_clusters(self) -> List[LogCluster]:
        """Return all discovered clusters."""
        return self.clusters
    
    # Registry-specific helper methods
    def _normalize_root_key(self, path: str) -> str:
        """Standardize Registry Root Key."""
        if not path or not isinstance(path, str):
            return '<ROOT:UNKNOWN>'
        
        path_upper = path.upper()
        root_mapping = {
            'HKLM': '<ROOT:HKLM>', 'HKEY_LOCAL_MACHINE': '<ROOT:HKLM>',
            'HKCU': '<ROOT:HKCU>', 'HKEY_CURRENT_USER': '<ROOT:HKCU>',
            'HKCR': '<ROOT:HKCR>', 'HKEY_CLASSES_ROOT': '<ROOT:HKCR>',
            'HKU': '<ROOT:HKU>', 'HKEY_USERS': '<ROOT:HKU>',
            'HKCC': '<ROOT:HKCC>', 'HKEY_CURRENT_CONFIG': '<ROOT:HKCC>'
        }
        
        for prefix, tag in root_mapping.items():
            if path_upper.startswith(prefix):
                return tag
        
        return '<ROOT:OTHER>'
    
    def _categorize_operation(self, operation: str) -> str:
        """Categorize registry operations."""
        if not operation or not isinstance(operation, str):
            return '<OP:UNKNOWN>'
        
        if operation in self.QUERY_OPS:
            return '<OP:QUERY>'
        elif operation in self.WRITE_OPS:
            return '<OP:WRITE>'
        elif operation in self.DELETE_OPS:
            return '<OP:DELETE>'
        elif operation in self.CLOSE_OPS:
            return '<OP:CLOSE>'
        
        return f'<OP:{operation.upper()}>'
    
    def _extract_data_type(self, detail: str) -> str:
        """Extract registry data type (REG_SZ, REG_DWORD, etc.)."""
        if not detail or not isinstance(detail, str):
            return '<TYPE:NONE>'
        
        match = re.search(r'Type:\s*(REG_\w+)', detail, re.IGNORECASE)
        return f'<TYPE:{match.group(1).upper()}>' if match else '<TYPE:NONE>'
    
    def _extract_result_status(self, result: str) -> str:
        """Categorize registry operation results."""
        if not result or not isinstance(result, str):
            return '<RESULT:UNKNOWN>'
        
        result_upper = result.upper()
        status_mapping = {
            'SUCCESS': '<RESULT:SUCCESS>',
            'NOT FOUND': '<RESULT:NOT_FOUND>',
            'NAME NOT FOUND': '<RESULT:NOT_FOUND>',
            'ACCESS DENIED': '<RESULT:ACCESS_DENIED>',
            'DENIED': '<RESULT:ACCESS_DENIED>',
            'REPARSE': '<RESULT:REPARSE>',
            'BUFFER': '<RESULT:BUFFER>'
        }
        
        for key, tag in status_mapping.items():
            if key in result_upper:
                return tag
        
        return f'<RESULT:{result_upper}>'
    
    def _extract_subpath(self, path: str) -> str:
        """Extract registry subpath after root key."""
        if not path or not isinstance(path, str):
            return ''
        
        path_upper = path.upper()
        prefixes = ['HKEY_LOCAL_MACHINE\\', 'HKLM\\', 'HKEY_CURRENT_USER\\', 'HKCU\\',
                    'HKEY_CLASSES_ROOT\\', 'HKCR\\', 'HKEY_USERS\\', 'HKU\\',
                    'HKEY_CURRENT_CONFIG\\', 'HKCC\\']
        
        for prefix in prefixes:
            if path_upper.startswith(prefix):
                return path[len(prefix):]
        
        return path
    
    def parse_registry_event(self, operation: str = "", path: str = "", 
                             result: str = "", detail: str = "") -> Tuple[str, List[str]]:
        """Parse Procmon registry event with structural categorization."""
        # Convert all inputs to strings
        operation = str(operation) if operation and not isinstance(operation, str) else (operation or "")
        path = str(path) if path and not isinstance(path, str) else (path or "")
        result = str(result) if result and not isinstance(result, str) else (result or "")
        detail = str(detail) if detail and not isinstance(detail, str) else (detail or "")
        
        # Build structured message with categorical tags
        parts = [
            self._normalize_root_key(path),
            self._categorize_operation(operation),
            self._extract_result_status(result)
        ]
        
        type_token = self._extract_data_type(detail)
        if type_token != '<TYPE:NONE>':
            parts.append(type_token)
        
        subpath = self._extract_subpath(path)
        if subpath:
            parts.append(subpath)
        
        if detail:
            parts.append(detail)
        
        # Get template using standard parsing
        template, _ = self.parse(' '.join(str(p) for p in parts))
        
        # Return original fields as parameters (preserves complete context)
        params = [f for f in [operation, path, result, detail] if f]
        
        return template, params
    
    def parse_from_row(self, row_dict: dict) -> Tuple[str, List[str]]:
        """Parse from dictionary with Procmon columns (Operation, Path, Result, Detail)."""
        return self.parse_registry_event(
            operation=row_dict.get('Operation', ''),
            path=row_dict.get('Path', ''),
            result=row_dict.get('Result', ''),
            detail=row_dict.get('Detail', '')
        )


# Compatibility alias for registry parsing
class RegistryDrainParser(DrainParser):
    """Alias for DrainParser with registry mode enabled."""
    def __init__(self, **kwargs):
        kwargs.setdefault('depth', 6)
        kwargs['registry_mode'] = True
        super().__init__(**kwargs)
