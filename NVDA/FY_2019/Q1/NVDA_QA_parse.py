from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
import json
import os
from openai import OpenAI
import tiktoken
from datetime import datetime

class Analyst(BaseModel):
    name: str
    firm: str
    note: Optional[str] = None

class Response(BaseModel):
    respondent: str
    answer: str

class QAExchange(BaseModel):
    analyst: Analyst
    question: str
    response: Union[Response, List[Response]]

class EarningsQA(BaseModel):
    event_type: str = "earnings_call"
    company: str
    executives: Dict[str, str]
    qa_session: List[QAExchange]
    closing_remarks: Optional[Dict[str, str]] = None

class QAParser:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def _create_system_prompt(self) -> str:
        return """You are a financial analyst expert in parsing earnings call transcripts.
        Your task is to:
        1. Identify each Q&A exchange from the raw transcript
        2. Extract the analyst's name, firm, and any notes (e.g., "calling in for someone")
        3. Extract the complete question, preserving all context
        4. Extract all responses, maintaining speaker attribution
        5. Identify any closing remarks
        
        Format everything precisely according to the provided JSON schema.
        Ensure responses are properly attributed to executives.
        Handle cases where multiple executives respond to a single question."""

    def extract_raw_qa(self, transcript: str) -> str:
        """Extract Q&A portion from full transcript using OpenAI"""
        prompt = """Extract only the Q&A portion from this earnings call transcript. 
        Start from the first analyst question and end with the closing remarks.
        Keep the full text of questions and answers intact."""
        
        response = self.client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": "You are a financial transcript analyst."},
                {"role": "user", "content": prompt + "\n\nTranscript:\n" + transcript}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content

    def structure_qa(self, qa_text: str, company_name: str) -> EarningsQA:
        """Use OpenAI to structure the Q&A text into our format"""
        schema_prompt = f"""Convert this Q&A transcript into a structured JSON format with the following requirements:

        1. Identify executives from the transcript
        2. For each Q&A exchange:
           - Extract analyst name and firm
           - Note if they're calling in for someone else
           - Preserve the full question
           - Include full response(s) with speaker attribution
           - Handle multiple responses to a single question
        3. Include any closing remarks

        The JSON should match this structure:
        {json.dumps(EarningsQA.model_json_schema(), indent=2)}

        Company name: {company_name}

        Transcript:
        {qa_text}"""

        response = self.client.chat.completions.create(
            model="gpt-4o-2024-11-20",
            messages=[
                {"role": "system", "content": self._create_system_prompt()},
                {"role": "user", "content": schema_prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        try:
            parsed_data = json.loads(response.choices[0].message.content)
            return EarningsQA(**parsed_data)
        except Exception as e:
            print(f"Error parsing OpenAI response: {e}")
            raise

    def parse_transcript(self, transcript: str, company_name: str) -> EarningsQA:
        """Main method to parse transcript using OpenAI"""
        # First, extract just the Q&A portion
        qa_text = self.extract_raw_qa(transcript)
        
        # Then structure it into our format
        return self.structure_qa(qa_text, company_name)

    def parse_from_file(self, file_path: str, company_name: str) -> EarningsQA:
        """Parse transcript from a file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            transcript = file.read()
        return self.parse_transcript(transcript, company_name)

    def save_to_json(self, qa_report: EarningsQA, output_file: str) -> str:
        """Save the parsed Q&A report to a JSON file"""
        os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
        
        report_dict = qa_report.model_dump()
        report_dict["metadata"] = {
            "parsed_date": datetime.now().isoformat(),
            "source": "earnings_call_qa_transcript"
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report_dict, indent=2, ensure_ascii=False, fp=f)
        
        print(f"Successfully saved parsed Q&A report to: {output_file}")
        return output_file

def main():
    # Initialize parser with your OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OpenAI API key not found in environment variables")
    
    parser = QAParser(api_key)
    
    # Input and output file paths
    input_file = "2018-05-10_earnings_qa.json"  # Raw transcript
    output_file = "2018-05-10_earnings_qa_parsed.json"
    
    try:
        # Parse the transcript
        qa_report = parser.parse_from_file(input_file, "NVIDIA")
        
        # Save the parsed report
        output_path = parser.save_to_json(
            qa_report=qa_report,
            output_file=output_file
        )
        
        # Print the parsed report
        print("\nParsed Q&A Report Structure:")
        print(json.dumps(qa_report.model_dump(), indent=2))
            
    except Exception as e:
        print(f"Error processing transcript: {str(e)}")
        raise

if __name__ == "__main__":
    main()