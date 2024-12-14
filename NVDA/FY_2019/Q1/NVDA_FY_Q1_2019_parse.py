import json
import os
import logging
from typing import List, Optional, Dict, Union, Any
from pydantic import BaseModel, Field, field_validator
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductInfo(BaseModel):
    """Product specific information."""
    name: str
    details: Optional[str] = None
    availability: Optional[str] = None
    price: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = Field(default_factory=dict)

class GeographicPerformance(BaseModel):
    """Geographic performance metrics."""
    region: str
    revenue_contribution: float = 0.0
    growth_rate: float = 0.0
    key_markets: List[str] = Field(default_factory=list)
    notable_developments: List[str] = Field(default_factory=list)

    @field_validator('revenue_contribution', 'growth_rate')
    @classmethod
    def default_zero_if_none(cls, v):
        return 0.0 if v is None else v

class AIInitiative(BaseModel):
    """AI-related initiatives and partnerships."""
    name: str
    partner: Optional[str] = None
    description: str
    deployment_status: Optional[str] = None
    target_market: Optional[str] = None
    expected_impact: Optional[str] = None

class SegmentMetrics(BaseModel):
    """Enhanced base model for segment-specific metrics."""
    revenue: float = 0.0
    yoy_growth: float = 0.0
    sequential_growth: float = 0.0
    fiscal_year_revenue: float = 0.0
    fiscal_year_growth: float = 0.0
    key_products: List[ProductInfo] = Field(default_factory=list)
    customer_wins: List[str] = Field(default_factory=list)
    strategic_updates: List[str] = Field(default_factory=list)
    geographic_performance: List[GeographicPerformance] = Field(default_factory=list)
    margin_info: Dict[str, float] = Field(default_factory=dict)

    @field_validator('revenue', 'yoy_growth', 'sequential_growth', 'fiscal_year_revenue', 'fiscal_year_growth')
    @classmethod
    def default_zero_if_missing(cls, v):
        return 0.0 if v is None else v

class DataCenterMetrics(SegmentMetrics):
    """Enhanced Data Center segment specific metrics."""
    inference_percentage: float = 0.0
    cloud_provider_percentage: float = 0.0
    china_percentage: float = 0.0
    networking_revenue_run_rate: float = 0.0
    compute_growth: Optional[str] = None
    networking_growth: Optional[str] = None
    software_services_run_rate: float = 0.0
    customer_segments: Dict[str, List[str]] = Field(default_factory=dict)
    major_platforms: List[ProductInfo] = Field(default_factory=list)
    ai_partners: List[AIInitiative] = Field(default_factory=list)
    cloud_initiatives: List[Dict[str, str]] = Field(default_factory=list)
    inference_products: List[ProductInfo] = Field(default_factory=list)

    @field_validator('inference_percentage', 'cloud_provider_percentage', 'china_percentage', 
                    'networking_revenue_run_rate', 'software_services_run_rate')
    @classmethod
    def default_float_zero(cls, v):
        return 0.0 if v is None else v

class GamingMetrics(SegmentMetrics):
    """Enhanced Gaming segment specific metrics."""
    rtx_installed_base: int = 0
    ai_enabled_applications: int = 0
    new_products: List[ProductInfo] = Field(default_factory=list)
    platform_metrics: Dict[str, Any] = Field(default_factory=dict)
    cloud_gaming_stats: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('rtx_installed_base', 'ai_enabled_applications')
    @classmethod
    def default_int_zero(cls, v):
        return 0 if v is None else v

class IndustryVerticalInfo(BaseModel):
    """Detailed information about an industry vertical."""
    key_trends: str = ""
    customer_wins: List[str] = Field(default_factory=list)
    market_size: float = 0.0
    growth_rate: float = 0.0
    key_initiatives: List[str] = Field(default_factory=list)
    strategic_priorities: List[str] = Field(default_factory=list)

    @field_validator('market_size', 'growth_rate')
    @classmethod
    def default_float_zero(cls, v):
        return 0.0 if v is None else v

class AIMetrics(BaseModel):
    """AI-specific metrics and initiatives."""
    total_ai_customers: int = 0
    new_ai_customers: int = 0
    ai_revenue_percentage: float = 0.0
    key_ai_products: List[ProductInfo] = Field(default_factory=list)
    ai_partnerships: List[AIInitiative] = Field(default_factory=list)
    industry_adoption: Dict[str, List[str]] = Field(default_factory=dict)
    deployment_metrics: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('total_ai_customers', 'new_ai_customers')
    @classmethod
    def default_int_zero(cls, v):
        return 0 if v is None else v

    @field_validator('ai_revenue_percentage')
    @classmethod
    def default_float_zero_ai(cls, v):
        return 0.0 if v is None else v

class FinancialMetrics(BaseModel):
    """Enhanced comprehensive financial metrics."""
    quarter: str
    fiscal_year: str
    date: str
    revenue: float = 0.0
    revenue_yoy_growth: float = 0.0
    revenue_sequential_growth: float = 0.0
    fiscal_year_revenue: float = 0.0
    fiscal_year_growth: float = 0.0
    gross_margin_gaap: float = 0.0
    gross_margin_non_gaap: float = 0.0
    operating_margin_gaap: float = 0.0
    operating_margin_non_gaap: float = 0.0
    operating_expenses_gaap: float = 0.0
    operating_expenses_non_gaap: float = 0.0
    operating_expenses_growth_gaap: float = 0.0
    operating_expenses_growth_non_gaap: float = 0.0
    cash_flow: Dict[str, float] = Field(default_factory=dict)
    balance_sheet_highlights: Dict[str, float] = Field(default_factory=dict)
    share_repurchases: float = 0.0
    dividends: float = 0.0
    total_shareholder_returns: float = 0.0
    segment_breakdown: Dict[str, float] = Field(default_factory=dict)
    geographic_breakdown: Dict[str, float] = Field(default_factory=dict)

    @field_validator('revenue', 'revenue_yoy_growth', 'revenue_sequential_growth', 'fiscal_year_revenue',
                    'fiscal_year_growth', 'gross_margin_gaap', 'gross_margin_non_gaap', 'operating_margin_gaap',
                    'operating_margin_non_gaap', 'operating_expenses_gaap', 'operating_expenses_non_gaap',
                    'operating_expenses_growth_gaap', 'operating_expenses_growth_non_gaap', 'share_repurchases',
                    'dividends', 'total_shareholder_returns')
    @classmethod
    def default_zero_for_nums(cls, v):
        return 0.0 if v is None else v

    @field_validator('quarter', 'fiscal_year', 'date')
    @classmethod
    def default_empty_string(cls, v):
        return v or ""

class NvidiaEarningsReport(BaseModel):
    """Enhanced complete earnings report structure."""
    financial_metrics: FinancialMetrics
    data_center: DataCenterMetrics
    gaming: GamingMetrics
    professional_visualization: SegmentMetrics
    automotive: SegmentMetrics
    ai_metrics: AIMetrics = Field(default_factory=AIMetrics)
    industry_verticals: Dict[str, IndustryVerticalInfo] = Field(default_factory=dict)
    guidance: Dict[str, Any] = Field(default_factory=dict)
    strategic_initiatives: List[str] = Field(default_factory=list)
    market_trends: List[str] = Field(default_factory=list)
    technology_roadmap: List[str] = Field(default_factory=list)
    competitive_positioning: List[str] = Field(default_factory=list)
    qa_section: List[Dict[str, Any]] = Field(default_factory=list)
def create_default_report() -> NvidiaEarningsReport:
    """Create a default report with all required fields initialized."""
    return NvidiaEarningsReport(
        financial_metrics=FinancialMetrics(
            quarter="",
            fiscal_year="",
            date="",
            revenue=0.0,
            gross_margin_gaap=0.0,
            gross_margin_non_gaap=0.0,
            operating_margin_gaap=0.0,
            operating_margin_non_gaap=0.0,
            operating_expenses_gaap=0.0,
            operating_expenses_non_gaap=0.0,
            operating_expenses_growth_gaap=0.0,
            operating_expenses_growth_non_gaap=0.0,
            share_repurchases=0.0,
            total_shareholder_returns=0.0
        ),
        data_center=DataCenterMetrics(),
        gaming=GamingMetrics(),
        professional_visualization=SegmentMetrics(),
        automotive=SegmentMetrics()
    )

PARSING_PROMPT = """
You are an expert financial analyst specializing in NVIDIA and semiconductor companies. 
Parse the following earnings call transcript with extreme attention to detail and granular data extraction.

Important: Return ONLY the JSON output without any additional text or markdown formatting.

Focus Areas (Ignore Q&A for now):
1. Financial Metrics:
   - ALL revenue figures (total, segment-wise, growth rates)
   - ALL margin figures (GAAP and non-GAAP)
   - Detailed segment performance metrics
   - Geographic breakdowns where mentioned
   - Cash flow and balance sheet highlights

2. AI and Data Center:
   - Specific AI customer wins and use cases
   - Cloud provider partnerships and initiatives
   - Inference vs training details
   - Networking and software/services metrics
   - Data center product introductions and roadmaps

3. Gaming and Professional Visualization:
   - New product launches, architectures, and specifications
   - Platform metrics and adoption rates
   - Cloud gaming initiatives
   - RTX and other GPU architecture updates

4. Industry Vertical Analysis:
   - Vertical-wise breakdown and metrics
   - Customer wins, market trends, and opportunities
   - Strategic initiatives and priorities

5. Forward-Looking Information:
   - Detailed guidance metrics
   - Technology roadmap
   - Strategic initiatives
   - Market outlook
   - Competitive positioning

For numeric fields not explicitly mentioned, default to 0.0 (no null/None).
Return ONLY the final JSON.

Transcript:
{transcript}

{format_instructions}
"""

def parse_earnings_transcript(transcript_text: str, model: str = "gpt-4o-2024-11-20") -> NvidiaEarningsReport:
    """Parse entire earnings transcript using a single LLM call with enhanced granularity."""
    try:
        llm = ChatOpenAI(
            model=model,
            temperature=0
        )

        parser = PydanticOutputParser(pydantic_object=NvidiaEarningsReport)

        prompt = PromptTemplate(
            template=PARSING_PROMPT,
            input_variables=["transcript"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )

        messages = [HumanMessage(content=prompt.format(transcript=transcript_text))]
        output = llm.invoke(messages).content

        logger.info("Successfully processed transcript with LLM")
        logger.debug(f"LLM Output: {output}")

        try:
            # Extract JSON content if wrapped in code fences
            if "```json" in output:
                json_content = output.split("```json")[1].split("```")[0].strip()
            else:
                json_content = output.strip()

            parsed_transcript = parser.parse(json_content)
            return parsed_transcript

        except Exception as parse_error:
            logger.error(f"Parsing error: {str(parse_error)}")
            return create_default_report()

    except Exception as e:
        logger.error(f"Error in transcript processing: {str(e)}")
        return create_default_report()

def main():
    """Main execution function."""
    try:
        file_path = "2018-05-10_earnings_call.json"
        logger.info(f"Processing transcript from {file_path}")

        # Load transcript
        with open(file_path, 'r') as f:
            raw_data = json.load(f)

        transcript = raw_data.get('content', '')

        # Parse transcript
        parsed_data = parse_earnings_transcript(transcript)

        # Save results
        output_file = "parsed_earnings_call.json"
        with open(output_file, 'w') as f:
            json.dump(parsed_data.dict(), f, indent=2)

        logger.info(f"Successfully saved parsed data to: {output_file}")

    except Exception as e:
        logger.error(f"Script execution failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
