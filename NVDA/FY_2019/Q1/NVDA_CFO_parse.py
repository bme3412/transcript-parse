from typing import List, Optional, Dict
from pydantic import BaseModel, Field
import json
import os
from openai import OpenAI
import tiktoken
from datetime import datetime

class FinancialMetric(BaseModel):
    value: float
    unit: str = Field(description="The unit of the metric (e.g., billion, million, percentage)")
    context: str = Field(description="Brief context about what this number represents")
    period: str = Field(description="Time period this metric relates to (e.g., Q1, Full Year)")

class ProductSegment(BaseModel):
    name: str
    revenue: Optional[FinancialMetric]
    growth_yoy: Optional[float]
    growth_qoq: Optional[float]
    key_points: List[str] = Field(description="Key highlights or developments for this segment")

class OperationalMetrics(BaseModel):
    gross_margin: Optional[FinancialMetric]
    operating_margin: Optional[FinancialMetric]
    net_income: Optional[FinancialMetric]
    cash_flow: Optional[FinancialMetric]
    other_metrics: Dict[str, FinancialMetric] = Field(description="Other important operational metrics mentioned")

class EarningsReport(BaseModel):
    company_name: str
    quarter: str
    fiscal_year: str
    total_revenue: FinancialMetric
    segments: List[ProductSegment]
    operational_metrics: OperationalMetrics
    future_guidance: Dict[str, FinancialMetric]
    strategic_initiatives: List[str] = Field(description="Key strategic initiatives or announcements")
    market_opportunities: List[Dict[str, str]] = Field(description="New market opportunities mentioned")

class EarningsCallParser:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken"""
        return len(self.encoding.encode(text))
    
    def _create_system_prompt(self) -> str:
        return """You are a financial analyst expert in parsing earnings call transcripts.
        Extract information precisely and structure it according to the provided format.
        Be exact with numbers and provide full context for each metric."""
    
    def _create_parsing_prompt(self, text: str) -> str:
        return f"""Parse the following earnings call transcript segment and extract:
        1. All financial metrics with exact values and units
        2. Segment performance details
        3. Growth rates (YoY and QoQ)
        4. Strategic initiatives
        5. Market opportunities

        Format the response as a valid JSON object matching this Pydantic model:
        {json.dumps(EarningsReport.model_json_schema(), indent=2)}

        Transcript:
        {text}"""

    def _chunk_text(self, text: str, max_tokens: int = 4000) -> List[str]:
        """Split text into chunks that fit within token limits"""
        chunks = []
        current_chunk = []
        current_length = 0
        
        sentences = text.split('. ')
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            if current_length + sentence_tokens > max_tokens:
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
            response = self.client.chat.completions.create(
                model="gpt-4o-2024-11-20",  # You can also use "gpt-3.5-turbo" for lower cost
                messages=[
                    {"role": "system", "content": self._create_system_prompt()},
                    {"role": "user", "content": self._create_parsing_prompt(chunk)}
                ],
                temperature=0.1,
                max_tokens=4000,
                response_format={"type": "json_object"}
            )
            
            try:
                parsed_chunk = json.loads(response.choices[0].message.content)
                parsed_chunks.append(parsed_chunk)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON response: {e}")
                continue
        
        # Merge parsed chunks into a single report
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
        """Parse transcript from a JSON file"""
        with open(json_file_path, 'r') as file:
            data = json.load(file)
            if 'CFO' in data:
                return self.parse_transcript(data['CFO'])
            raise ValueError("Expected JSON structure not found")

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
        
        # Convert to dict using the new model_dump method
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
    # Initialize parser with your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    parser = EarningsCallParser(api_key)
    
    # Parse from JSON file
    json_file_path = "2018-05-10_earnings_CFO.json"
    try:
        # Parse the transcript
        earnings_report = parser.parse_from_json(json_file_path)
        
        # Save the parsed report to a JSON file
        output_path = parser.save_to_json(
            earnings_report=earnings_report,
            company="NVDA"
        )
        
        # Print the parsed report to console
        print("\nParsed Earnings Report Structure:")
        print(json.dumps(earnings_report.model_dump(), indent=2))
            
    except Exception as e:
        print(f"Error parsing transcript: {str(e)}")

if __name__ == "__main__":
    main()