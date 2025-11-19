import asyncio
import os
from typing import List, Dict, Any, Optional

# --- Mocking the Agent Framework Dependencies ---
# In a real environment, you would replace these with your actual framework imports (e.g., from your Agents SDK).

class BaseModel:
    """Mock Pydantic BaseModel."""
    def __init__(self, **data: Any):
        for key, value in data.items():
            setattr(self, key, value)
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.__dict__})"
        
    def __str__(self) -> str:
        return self.__repr__()
        
    def dict(self):
        return self.__dict__

def Field(*args, **kwargs):
    """Mock Pydantic Field function."""
    return None

class MockModel:
    """Mock for the Gemini model instance (e.g., gemini-2.0-flash)."""
    pass

# Mock instance of the model
gemini_model = MockModel() 

class Agent:
    """Mock Agent class."""
    def __init__(self, name: str, instructions: str, model: MockModel, output_type: Optional[Any] = None, tools: Optional[List] = None, model_settings: Optional[Any] = None):
        self.name = name
        self.instructions = instructions
        self.output_type = output_type
        self.tools = tools
        self.model_settings = model_settings
        
class RunnerResult:
    """Mock result from the Runner."""
    def __init__(self, final_output: Any):
        self.final_output = final_output

class Runner:
    """Mock Runner class to simulate agent execution."""
    
    # Store the query history for the mock logic
    query_history = []
    
    @staticmethod
    async def run(agent: Agent, input_data: Any) -> RunnerResult:
        """Simulates agent execution based on the agent's name."""
        
        print(f"\n[Mock Runner: Executing {agent.name}]")
        # print(f"  Input length: {len(str(input_data))}") # Too noisy
        
        # --- 1. Research Scout Logic (Planning) ---
        if agent.name == "Research Scout":
            # Initial plan based on the Mapper Agent's graph
            mock_output = ResearchScoutOutput(
                search_queries=[
                    SearchQueryItem(
                        focus_entity="Economic Viability", 
                        query="Cost savings and ROI of four-day work week pilot programs"
                    ),
                    SearchQueryItem(
                        focus_entity="Productivity Impact", 
                        query="Academic studies on employee productivity in compressed work schedules"
                    ),
                    SearchQueryItem(
                        focus_entity="Legal/Regulatory Status", 
                        query="Current legislation regarding 4-day work week in the US and EU"
                    ),
                ]
            )
            return RunnerResult(mock_output)

        # --- 2. Search Agent Logic (Inside _execute_single_search) ---
        elif agent.name == "Search Agent":
            # Always returns a simple, summarized result
            query = input_data.split('\n')[0].replace('Search item: ', '')
            
            summary = (
                f"Summary for: '{query[:50]}...'. "
                "Initial reports show mixed results; some UK trials report 30% revenue increases, while a Utah program noted some customer service issues. "
                "Productivity measurement remains inconsistent across studies. Legal status is largely unregulated, relying on existing wage and hour laws."
            )
            return RunnerResult(summary)

        # --- 3. Critique Agent Logic (Self-Correction Loop) ---
        elif agent.name == "Critique Agent":
            # Use query history to simulate iterations
            # If the number of executed searches is 3 or less (i.e., just the initial batch)
            if len([q for q in Runner.query_history if q != 'critique_run']) <= 3:
                # --- Iteration 1: Force a gap to trigger refinement ---
                print("  (Mock Logic: Triggering Gaps Found to force iteration 2)")
                
                mock_output = CritiqueAgentOutput(
                    gaps_found=True,
                    critique_summary="The initial results are too broad, lacking specific country-level labor data and comparative analysis against standard 5-day models. The financial data is anecdotal.",
                    refined_queries=[
                        RefinedQuery(
                            reason_for_refinement="Need specific data on labor law compliance in Germany/UK.",
                            new_query="Impact of 4-day work week on German labor union contracts"
                        ),
                        RefinedQuery(
                            reason_for_refinement="Need specific, recent Q3/Q4 2024 financial reports from companies using 4-day models.",
                            new_query="Q3 2024 earnings report companies using four-day work week"
                        ),
                    ]
                )
            else:
                # --- Iteration 2+: Assume successful verification ---
                print("  (Mock Logic: Verification Successful, stopping loop)")
                
                mock_output = CritiqueAgentOutput(
                    gaps_found=False,
                    critique_summary="All initial claims are now sufficiently supported by targeted, recent research, including German labor data and specific Q3 financial reports.",
                    refined_queries=[]
                )
            Runner.query_history.append("critique_run")
            return RunnerResult(mock_output)
        
        return RunnerResult("Mock output not configured for this agent.")

# --- Structured Models ---

class SearchQueryItem(BaseModel):
    """A single, highly specific search query."""
    
    focus_entity: str = Field(description="The key entity or claim this search query is designed to investigate (e.g., 'four-day work week economic viability').")
    
    query: str = Field(description="The specific keyword or question for the search engine.")
    
class ResearchScoutOutput(BaseModel):
    """A collection of generated search queries."""
    
    search_queries: List[SearchQueryItem] = Field(
        description="A list of specific search queries formulated to investigate the concepts and claims from the Draft Knowledge Graph."
    )

class RefinedQuery(BaseModel):
    """A new or refined query generated to address a gap in the current evidence."""
    
    reason_for_refinement: str = Field(description="Explains why the current evidence is insufficient (e.g., 'Sources were too old', 'Contradictory data found', 'Missing financial data').")
    
    new_query: str = Field(description="A highly focused, specific query to fill the identified gap.")
    
class CritiqueAgentOutput(BaseModel):
    """The structured report on the quality of the current research results."""
    
    gaps_found: bool = Field(description="True if the current raw results are incomplete, contradictory, or lack sufficient factual support. This triggers the self-correction loop.")
    
    critique_summary: str = Field(description="A brief summary of the overall evidence quality and the primary missing pieces.")
    
    refined_queries: List[RefinedQuery] = Field(
        description="A list of new queries to be executed if gaps_found is True."
    )

class MapperAgentOutput(BaseModel):
    """Mocking the input structure from the previous stage."""
    draft_knowledge_graph: str = Field(description="Conceptual graph of claims and concepts.")

# --- Agent Definitions ---

SCOUT_AGENT_INSTRUCTIONS = "..." 
CRITIQUE_AGENT_INSTRUCTIONS = "..."
SEARCH_AGENT_INSTRUCTIONS = "..."

search_agent = Agent(name="Search Agent", instructions=SEARCH_AGENT_INSTRUCTIONS, model=gemini_model)
research_scout_agent = Agent(name="Research Scout", instructions=SCOUT_AGENT_INSTRUCTIONS, model=gemini_model, output_type=ResearchScoutOutput)
critique_agent = Agent(name="Critique Agent", instructions=CRITIQUE_AGENT_INSTRUCTIONS, model=gemini_model, output_type=CritiqueAgentOutput)


# --- Core Orchestration and Execution Logic ---

async def _execute_single_search(item: SearchQueryItem):
    """Internal helper to execute a single search using the Search Agent."""
    
    # Store the query in mock history for critique logic
    Runner.query_history.append(item.query) 
    
    input_data = f"Search item: {item.query}\nFocus: {item.focus_entity}"
    result = await Runner.run(search_agent, input_data)
    
    # Return the query and the summarized result together for the Critique Agent
    return {
        "query": item.query,
        "focus": item.focus_entity,
        "summary": result.final_output # This is the 200-word summary from Search Agent
    }

async def run_research_crew(draft_graph_output: MapperAgentOutput) -> List[Dict]:
    """
    The Orchestrator function. Manages the self-correction loop.
    Input: The structured output from the Mapper Agent.
    Output: A verified list of search result summaries.
    """
    
    print("\n\n#############################################")
    print("## STARTING RESEARCH ORCHESTRATOR (Stage 2) ##")
    print("#############################################")
    
    # Reset mock history for a fresh run
    Runner.query_history = []
    
    # 1. Initial Planning (Scout)
    print("\n[STEP 1/4: Planning] Running Research Scout for Initial Plan...")
    scout_plan_result = await Runner.run(research_scout_agent, draft_graph_output.draft_knowledge_graph)
    current_queries: List[SearchQueryItem] = scout_plan_result.final_output.search_queries
    
    MAX_ITERATIONS = 3
    all_raw_results = []
    
    for iteration in range(MAX_ITERATIONS):
        print(f"\n=============================================")
        print(f"       RESEARCH ITERATION {iteration + 1} / {MAX_ITERATIONS}")
        print(f"=============================================")
        print(f"Queries to execute in this iteration (Count: {len(current_queries)}):")
        for i, q in enumerate(current_queries):
            print(f"  {i+1}. [{q.focus_entity}] -> {q.query}")
            
        # 2. Execute Searches Concurrently
        print("\n[STEP 2/4: Execution] Executing search queries concurrently...")
        tasks = [asyncio.create_task(_execute_single_search(q)) for q in current_queries]
        raw_results = await asyncio.gather(*tasks)
        
        # Accumulate results
        new_results_count = len(raw_results)
        all_raw_results.extend(raw_results)
        print(f"  -> Successfully executed {new_results_count} searches.")
        print(f"  -> Total accumulated results: {len(all_raw_results)}")
        
        # Consolidate ALL results for the Critique Agent's input
        results_text = "\n\n".join([
            f"QUERY: {r['query']}\nFOCUS: {r['focus']}\nSUMMARY: {r['summary']}" 
            for r in all_raw_results
        ])
        
        # The Critique Agent needs the claims and ALL current results
        critique_input = (
            f"CLAIMS TO VALIDATE: {draft_graph_output.draft_knowledge_graph}\n\n"
            f"ALL ACCUMULATED RAW RESULTS ({len(all_raw_results)} total):\n{results_text}"
        )
        
        # 3. Run the Critique Agent
        print("\n[STEP 3/4: Verification] Running Critique Agent with accumulated data.")
        print(f"  -> Critique Agent input size: {len(critique_input)} characters.")
        
        critique_result = await Runner.run(critique_agent, critique_input)
        critique_report: CritiqueAgentOutput = critique_result.final_output
        
        print("\n--- CRITIQUE REPORT ---")
        print(f"  Gaps Found (gaps_found): {critique_report.gaps_found}")
        print(f"  Critique Summary: {critique_report.critique_summary}")

        # 4. Check for Gaps (Self-Correction Logic)
        if not critique_report.gaps_found:
            print("\n✅ Verification successful: Evidence is sufficient. Halting research loop.")
            return all_raw_results
        
        # If gaps found, prepare for next iteration
        refined_queries_count = len(critique_report.refined_queries)
        print(f"\n🔄 Gaps found. Preparing {refined_queries_count} refined queries for ITERATION {iteration + 2}.")
        
        current_queries = [
            SearchQueryItem(focus_entity=q.reason_for_refinement, query=q.new_query)
            for q in critique_report.refined_queries
        ]

    # Fallback if max iterations reached
    print("\n\n🛑 WARNING: Max research iterations reached. Returning accumulated results.")
    return all_raw_results


# ---------------------------------------------
# --- Test Execution Block (Use this in your notebook) ---
# ---------------------------------------------

# 1. Define the input the user requested
INITIAL_MAPPER_OUTPUT = MapperAgentOutput(
    draft_knowledge_graph=(
        "Four-day Work Week [IMPLIES] Increased Employee Well-being. "
        "Four-day Work Week [INQUIRES] Economic Viability. "
        "Economic Viability [DEPENDS_ON] Employee Productivity. "
        "Increased Employee Well-being [LEADS_TO] Lower Turnover Rate."
    )
)
print("--- Initial Mapper Agent Output Defined ---")
print(INITIAL_MAPPER_OUTPUT)


# 2. Define the execution function
async def execute_test():
    """Runs the main orchestrator function with the sample input."""
    # Start the crew
    final_results = await run_research_crew(INITIAL_MAPPER_OUTPUT)
    
    print("\n\n#############################################")
    print("## FINAL ACCUMULATED RESULTS ##")
    print("#############################################")
    print(f"Total VERIFIED search summaries collected: {len(final_results)}")
    
    # for i, res in enumerate(final_results):
    #     print(f"\n--- Result {i+1} ({res['focus']}) ---")
    #     print(f"Query: {res['query']}")
    #     print(f"Summary: {res['summary']}")

# 3. Run the test
if __name__ == "__main__":
    asyncio.run(execute_test())