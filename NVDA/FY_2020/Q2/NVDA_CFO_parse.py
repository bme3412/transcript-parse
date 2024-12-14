from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import json
import os
from openai import OpenAI
import tiktoken
from datetime import datetime

class FinancialMetric(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = Field(default=None, description="The unit of the metric (e.g., billion, million, percentage)")
    context: Optional[str] = Field(default=None, description="Brief context about what this number represents")
    period: Optional[str] = Field(default=None, description="Time period this metric relates to (e.g., Q1, Full Year)")

class ProductSegment(BaseModel):
    name: str
    revenue: Optional[FinancialMetric] = None
    growth_yoy: Optional[float] = None
    growth_qoq: Optional[float] = None
    key_points: List[str] = Field(default_factory=list, description="Key highlights or developments for this segment")

class OperationalMetrics(BaseModel):
    gross_margin: Optional[FinancialMetric] = None
    operating_margin: Optional[FinancialMetric] = None
    net_income: Optional[FinancialMetric] = None
    cash_flow: Optional[FinancialMetric] = None
    other_metrics: Dict[str, FinancialMetric] = Field(default_factory=dict, description="Other important operational metrics mentioned")

class EarningsReport(BaseModel):
    company_name: str = "NVIDIA"  # Default value
    quarter: str = "Q2"  # Default value
    fiscal_year: str = "2020"  # Default value
    total_revenue: Optional[FinancialMetric] = None
    segments: List[ProductSegment] = Field(default_factory=list)
    operational_metrics: OperationalMetrics = Field(default_factory=OperationalMetrics)
    future_guidance: Dict[str, FinancialMetric] = Field(default_factory=dict)
    strategic_initiatives: List[str] = Field(default_factory=list)
    market_opportunities: List[Dict[str, str]] = Field(default_factory=list)

class EarningsCallParser:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def _create_system_prompt(self) -> str:
        return """You are a financial analyst expert in parsing earnings call transcripts.
        Extract key financial metrics, segment performance, and strategic initiatives.
        When a metric is mentioned, always include its value, unit, and context.
        For segments without explicit revenue numbers, leave the revenue field as null.
        Be precise and thorough in extracting all numerical data."""
    
    def _create_parsing_prompt(self, text: str) -> str:
        schema = EarningsReport.model_json_schema()
        # Remove potentially problematic default values from schema
        if 'properties' in schema:
            for prop in schema['properties'].values():
                if 'default' in prop:
                    del prop['default']
        
        return f"""Parse the following earnings call transcript segment and extract:
        1. All financial metrics with exact values and units
        2. Segment performance details (even if revenue is not explicitly stated)
        3. Growth rates (YoY and QoQ) where available
        4. Strategic initiatives
        5. Market opportunities

        Important notes:
        - Extract ALL numerical values mentioned with their proper context
        - Include segment-specific metrics and growth rates
        - Capture both current performance and forward-looking statements
        - Note any significant product launches or strategic developments
        
        Format the response as a valid JSON object matching this schema:
        {json.dumps(schema, indent=2)}

        Transcript:
        {text}"""

    def _chunk_text(self, text: str, max_tokens: int = 4000) -> List[str]:
        """Split text into chunks that fit within token limits"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Add a buffer to ensure we don't exceed token limits with prompt
        effective_max_tokens = max_tokens - 1000  # Buffer for prompt and overhead
        
        sentences = text.split('. ')
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            if current_length + sentence_tokens > effective_max_tokens:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_length = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_length += sentence_tokens
                
        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')
        return chunks

    def parse_transcript(self, transcript: str) -> EarningsReport:
        """Parse the earnings call transcript using OpenAI's API"""
        chunks = self._chunk_text(transcript)
        parsed_chunks = []
        
        for chunk in chunks:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-2024-11-20",  # Using standard GPT-4 model
                    messages=[
                        {"role": "system", "content": self._create_system_prompt()},
                        {"role": "user", "content": self._create_parsing_prompt(chunk)}
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                    response_format={"type": "json_object"}
                )
                
                parsed_chunk = json.loads(response.choices[0].message.content)
                parsed_chunks.append(parsed_chunk)
            except Exception as e:
                print(f"Error processing chunk: {str(e)}")
                continue
        
        if not parsed_chunks:
            return EarningsReport()  # Return empty report if all chunks failed
            
        return self._merge_parsed_chunks(parsed_chunks)
    
    def _merge_parsed_chunks(self, chunks: List[dict]) -> EarningsReport:
        """Merge multiple parsed chunks into a single EarningsReport"""
        if not chunks:
            raise ValueError("No parsed chunks to merge")
        
        merged = chunks[0]
        for chunk in chunks[1:]:
            # Merge segments
            merged['segments'].extend(chunk.get('segments', []))
            
            # Merge strategic initiatives
            merged['strategic_initiatives'].extend(chunk.get('strategic_initiatives', []))
            
            # Merge market opportunities
            merged['market_opportunities'].extend(chunk.get('market_opportunities', []))
            
            # Update metrics if they're more recent/complete
            if chunk.get('operational_metrics'):
                for key, value in chunk['operational_metrics'].items():
                    if value and not merged['operational_metrics'].get(key):
                        merged['operational_metrics'][key] = value
        
        # Remove duplicates
        merged['strategic_initiatives'] = list(set(merged['strategic_initiatives']))
        merged['segments'] = self._deduplicate_segments(merged['segments'])
        
        return EarningsReport(**merged)
    
    def _deduplicate_segments(self, segments: List[dict]) -> List[dict]:
        """Remove duplicate segments based on name"""
        seen = {}
        unique_segments = []
        for segment in segments:
            if segment['name'] not in seen:
                seen[segment['name']] = True
                unique_segments.append(segment)
        return unique_segments

    def parse_from_json(self, json_file_path: str) -> EarningsReport:
        """Parse transcript from a JSON file with better error handling"""
        try:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
                if 'CFO' in data:
                    parsed_report = self.parse_transcript(data['CFO'])
                    # Ensure all required fields have at least default values
                    if not parsed_report.total_revenue:
                        parsed_report.total_revenue = FinancialMetric()
                    if not parsed_report.operational_metrics:
                        parsed_report.operational_metrics = OperationalMetrics()
                    return parsed_report
                raise ValueError("Expected 'CFO' key not found in JSON")
        except Exception as e:
            print(f"Error reading or parsing JSON file: {str(e)}")
            # Return a minimal valid report rather than failing
            return EarningsReport(
                segments=[],
                operational_metrics=OperationalMetrics(),
                future_guidance={},
                strategic_initiatives=[],
                market_opportunities=[]
            )

    def save_to_json(self, earnings_report: EarningsReport, company: str, output_dir: str = "parsed_earnings") -> str:
        """
        Save the parsed earnings report to a JSON file with a structured filename
        
        Args:
            earnings_report: The parsed EarningsReport object
            company: Company ticker or name
            output_dir: Directory to save the JSON files
        
        Returns:
            str: Path to the saved JSON file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with company, quarter, and fiscal year
        filename = f"{company}_{earnings_report.fiscal_year}_{earnings_report.quarter}_earnings.json"
        filepath = os.path.join(output_dir, filename)
        
        # Convert to dict using the model_dump method
        report_dict = earnings_report.model_dump()
        
        # Add metadata
        report_dict["metadata"] = {
            "parsed_date": datetime.now().isoformat(),
            "company_ticker": company,
            "source": "earnings_call_transcript"
        }
        
        # Save to JSON file with proper formatting
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report_dict, indent=2, ensure_ascii=False, fp=f)
        
        print(f"Successfully saved parsed earnings report to: {filepath}")
        return filepath

def main():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    parser = EarningsCallParser(api_key)
    
    try:
        json_file_path = "NVDA_FY2020_Q2_CFO.json"
        earnings_report = parser.parse_from_json(json_file_path)
        
        output_path = parser.save_to_json(
            earnings_report=earnings_report,
            company="NVDA"
        )
        
        print("\nParsed Earnings Report Structure:")
        print(json.dumps(earnings_report.model_dump(), indent=2))
            
    except Exception as e:
        print(f"Error parsing transcript: {str(e)}")

if __name__ == "__main__":
    main()