from deepeval.metrics import BiasMetric, ToxicityMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
import re
from typing import Any, Dict, Type, Union
from pydantic import BaseModel, ValidationError

class QueryValidator:
    def __init__(self, prompt):
        self.prompt = prompt
        
    def validate(self):
        """Main validation method that runs all checks"""
        results = {}
        results['prompt_injection'] = self._check_prompt_injection()
        results['privacy'] = self._check_privacy()
        return results
        
    def _check_prompt_injection(self):
        """Check for potential prompt injection attacks"""
        injection_patterns = [
            r'ignore\s+previous\s+instructions',
            r'forget\s+your\s+system\s+prompt',
            r'you\s+are\s+now\s+a\s+different\s+character',
            r'disregard\s+the\s+above',
            r'override\s+your\s+programming',
            r'act\s+as\s+if\s+you\s+are',
            r'pretend\s+to\s+be',
            r'roleplay\s+as',
            r'simulate\s+being',
            r'bypass\s+your\s+guidelines'
        ]
        
        prompt_lower = self.prompt.lower()
        detected_patterns = []
        
        for pattern in injection_patterns:
            if re.search(pattern, prompt_lower, re.IGNORECASE):
                detected_patterns.append(pattern)
        
        return {
            'is_safe': len(detected_patterns) == 0,
            'detected_patterns': detected_patterns,
            'risk_level': 'high' if len(detected_patterns) > 2 else 'medium' if len(detected_patterns) > 0 else 'low'
        }
    
    def _check_privacy(self):
        """Check for potential privacy violations in the prompt"""
        privacy_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
            r'\b(?:password|secret|token|key)\s*[:=]\s*\S+',  # Credentials
        ]
        
        detected_pii = []
        for pattern in privacy_patterns:
            matches = re.findall(pattern, self.prompt, re.IGNORECASE)
            if matches:
                detected_pii.extend(matches)
        
        return {
            'is_safe': len(detected_pii) == 0,
            'detected_pii': detected_pii,
            'risk_level': 'high' if len(detected_pii) > 0 else 'low'
        }

class CompletionValidator:
    def __init__(self, completion, context=None):
        self.completion = completion
        self.context = context
        
    def validate(self):
        """Main validation method that runs all checks"""
        results = {}
        results['toxicity'] = self._check_toxicity()
        results['bias'] = self._check_bias()
        results['hallucination'] = self._check_hallucination()
        results['privacy'] = self._check_privacy()
        return results
        
    def _check_toxicity(self):
        """Check for toxic content using deepeval ToxicityMetric"""
        try:
            # Create a test case for toxicity evaluation
            test_case = LLMTestCase(
                input="",  # Not needed for toxicity check
                actual_output=self.completion
            )
            
            # Initialize toxicity metric with threshold
            toxicity_metric = ToxicityMetric(threshold=0.5)
            
            # Measure toxicity
            toxicity_metric.measure(test_case)
            
            return {
                'is_safe': toxicity_metric.success,
                'score': toxicity_metric.score,
                'reason': toxicity_metric.reason,
                'threshold': toxicity_metric.threshold
            }
        except Exception as e:
            return {
                'is_safe': True,  # Default to safe if evaluation fails
                'error': str(e),
                'score': 0.0
            }
    
    def _check_bias(self):
        """Check for bias using deepeval BiasMetric"""
        try:
            # Create a test case for bias evaluation
            test_case = LLMTestCase(
                input="",  # Not needed for bias check
                actual_output=self.completion
            )
            
            # Initialize bias metric with threshold
            bias_metric = BiasMetric(threshold=0.5)
            
            # Measure bias
            bias_metric.measure(test_case)
            
            return {
                'is_safe': bias_metric.success,
                'score': bias_metric.score,
                'reason': bias_metric.reason,
                'threshold': bias_metric.threshold
            }
        except Exception as e:
            return {
                'is_safe': True,  # Default to safe if evaluation fails
                'error': str(e),
                'score': 0.0
            }
    
    def _check_hallucination(self):
        """Check for hallucinations using deepeval HallucinationMetric"""
        try:
            if not self.context:
                return {
                    'is_safe': True,
                    'message': 'No context provided for hallucination check',
                    'score': 0.0
                }
            
            # Create a test case for hallucination evaluation
            test_case = LLMTestCase(
                input="",  # Not needed for hallucination check
                actual_output=self.completion,
                context=self.context if isinstance(self.context, list) else [self.context]
            )
            
            # Initialize hallucination metric with threshold
            hallucination_metric = HallucinationMetric(threshold=0.7)
            
            # Measure hallucination
            hallucination_metric.measure(test_case)
            
            return {
                'is_safe': hallucination_metric.success,
                'score': hallucination_metric.score,
                'reason': hallucination_metric.reason,
                'threshold': hallucination_metric.threshold
            }
        except Exception as e:
            return {
                'is_safe': True,  # Default to safe if evaluation fails
                'error': str(e),
                'score': 0.0
            }
    
    def _check_privacy(self):
        """Check for privacy violations in the completion"""
        privacy_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email
            r'\b\d{3}-\d{3}-\d{4}\b',  # Phone number
            r'\b(?:password|secret|token|key)\s*[:=]\s*\S+',  # Credentials
        ]
        
        detected_pii = []
        for pattern in privacy_patterns:
            matches = re.findall(pattern, self.completion, re.IGNORECASE)
            if matches:
                detected_pii.extend(matches)
        
        return {
            'is_safe': len(detected_pii) == 0,
            'detected_pii': detected_pii,
            'risk_level': 'high' if len(detected_pii) > 0 else 'low'
        }

class JSONValidator:
    def __init__(self, model: Type[BaseModel], data: Union[Dict[str, Any], Any]):
        """
        Initialize JSONValidator with a Pydantic model and data to validate.
        
        Args:
            model: A Pydantic BaseModel class to validate against
            data: The data to validate (dict, JSON string, or any object)
        """
        self.model = model
        self.data = data
        
    def validate(self):
        """
        Main validation method using Pydantic model validation.
        
        Returns:
            dict: Validation results with 'is_valid', 'errors', and 'validated_data' keys
        """
        return self._check_schema()
        
    def _check_schema(self):
        """
        Check if data matches the provided Pydantic model.
        
        Returns:
            dict: Validation results
        """
        try:
            # Validate data using Pydantic model
            validated_instance = self.model.model_validate(self.data)
            
            return {
                'is_valid': True,
                'errors': [],
                'validated_data': validated_instance.model_dump(),
                'model_name': self.model.__name__
            }
        except ValidationError as e:
            # Extract error details from Pydantic ValidationError
            errors = []
            for error in e.errors():
                error_detail = {
                    'field': '.'.join(str(loc) for loc in error['loc']) if error['loc'] else 'root',
                    'message': error['msg'],
                    'type': error['type'],
                    'input': error.get('input', 'N/A')
                }
                errors.append(error_detail)
            
            return {
                'is_valid': False,
                'errors': errors,
                'validated_data': None,
                'model_name': self.model.__name__,
                'raw_error': str(e)
            }
        except Exception as e:
            return {
                'is_valid': False,
                'errors': [{'field': 'root', 'message': f"Validation error: {str(e)}", 'type': 'unknown'}],
                'validated_data': None,
                'model_name': self.model.__name__ if hasattr(self, 'model') else 'Unknown',
                'raw_error': str(e)
            }
