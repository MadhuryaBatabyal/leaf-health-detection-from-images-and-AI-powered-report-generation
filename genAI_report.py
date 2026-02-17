import os
import json
from typing import Dict, Any, Optional

import google.generativeai as genai
from dotenv import load_dotenv


class GenAIReportGenerator:
    """
    Gemini-powered plant health report generator.

    Generates:
        1. Structured JSON analysis
        2. Detailed narrative explanation
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.3
    ):
        """
        Args:
            model_name: Gemini model to use
            temperature: creativity level (lower = more deterministic)
        """

        # Load environment variables from .env
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY not found. "
                "Make sure it is set in your .env file."
            )

        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(model_name)
        self.temperature = temperature

    # -------------------------------------------------------
    # Public Method
    # -------------------------------------------------------
    def generate(self, structured_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate hybrid plant health report.

        Args:
            structured_metrics: dict containing outputs from all modules

        Returns:
            {
                "structured_analysis": dict or None,
                "narrative_report": str
            }
        """

        prompt = self._build_prompt(structured_metrics)

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": self.temperature
            }
        )

        raw_text = response.text.strip()

        structured_output = self._extract_json(raw_text)

        return {
            "structured_analysis": structured_output,
            "narrative_report": raw_text
        }

    # -------------------------------------------------------
    # Prompt Builder
    # -------------------------------------------------------
    def _build_prompt(self, metrics: Dict[str, Any]) -> str:
        """
        Construct structured agronomic reasoning prompt.
        """

        return f"""
You are an expert agricultural advisor.

Analyze the following plant health metrics and generate a professional agronomic report.

Plant Metrics:
{json.dumps(metrics, indent=2)}

Instructions:
1. Explain what each metric biologically represents.
2. Interpret the current metric values.
3. Correlate disease, dryness, pests, and green index.
4. Determine overall plant health condition.
5. Provide actionable farmer recommendations.
6. Assign urgency level (Low, Moderate, High).

IMPORTANT:
- Base your reasoning ONLY on provided metrics.
- Do not hallucinate external data.
- Provide structured JSON first.
- After JSON, provide a detailed narrative explanation.

Return JSON in this exact format:

{{
  "overall_health": "...",
  "disease_analysis": "...",
  "dryness_analysis": "...",
  "green_index_analysis": "...",
  "pest_analysis": "...",
  "recommended_actions": ["...", "..."],
  "urgency": "Low/Moderate/High"
}}

After the JSON block, provide a detailed agronomic explanation.
"""

    # -------------------------------------------------------
    # JSON Extractor
    # -------------------------------------------------------
    def _extract_json(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Attempts to extract JSON block from Gemini output.
        """

        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            json_block = text[start:end]
            return json.loads(json_block)
        except Exception:
            return None
