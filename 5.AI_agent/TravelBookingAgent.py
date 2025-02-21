import warnings
warnings.filterwarnings('ignore')

from crewai_tools import SerperDevTool, ScrapeWebsiteTool, WebsiteSearchTool
from dotenv import load_dotenv, find_dotenv

load_dotenv()
# load tools
serper_search_tool = SerperDevTool()
scrap_tool = ScrapeWebsiteTool()
website_search_tool = WebsiteSearchTool()

from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gpt-3.5-turbo")

from crewai import Agent, Task, Crew
from crewai.process import Process

tools=[serper_search_tool, scrap_tool, website_search_tool]

# Define the Agents
travel_expert = Agent(
    role='Hotel Researcher',
    goal='Efficiently locate and compile a comprehensive list of suitable hotel options in the specified location, \
        adhering to the given criteria such as check-in/out dates, number of guests, and any other specific requirements. \
        Ensure the gathered information is accurate, up-to-date, and includes key details like pricing, amenities, and guest ratings.',
    backstory='You are an experienced digital travel concierge with a keen eye for detail and a passion for finding the perfect accommodations.\
        With years of experience in the hospitality industry and a vast knowledge of global hotel chains and boutique properties, you\
        have honed your skills in navigating various booking platforms and hotel databases. \
        Your expertise lies in quickly sifting through numerous options to identify the most suitable choices for travelers, \
        taking into account factors such as location, price, amenities, and guest reviews.',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=tools,
)

hotel_reviewer = Agent(
    role='Hotel Quality Analyst',
    goal='Thoroughly evaluate and compare hotel options to provide detailed, unbiased assessments that help travelers make informed decisions. \
        Analyze each hotel\'s amenities, location, value for money, and guest experiences to create comprehensive reviews that highlight strengths, weaknesses, and unique features.',
    backstory='You are a seasoned travel industry professional with over a decade of experience in hotel evaluation and critique. \
        Your background includes working as a luxury hotel inspector, a travel journalist for renowned publications, and a consultant for hotel rating systems. \
        This diverse experience has honed your ability to assess accommodations from multiple perspectives, considering both objective criteria and subjective guest experiences. \
        Your reviews are known for their depth, fairness, and ability to capture the essence of each property. You have a particular talent for identifying hidden gems and spotting potential issues that might affect a guest\'s stay. \
        Your expertise covers a wide range of accommodations, from budget-friendly options to ultra-luxury resorts, and you\'re adept at evaluating hotels in various cultural contexts around the world.',
    verbose=True,
    allow_delegation=False,
    llm=llm,
    tools=tools,
)
# Define the Tasks
hotel_search_task = Task(
    description="Search for 5 hotels in {location} for {number_of_people} adult people, checking in on {check_in} and checking out on {check_out}.",
    agent=travel_expert,
    expected_output="All the details of a specifically chosen accommodation.")

hotel_review_task = Task(
    description="Based on the recommendations provided, pick the best options based on ratings, reviews, and facilities available. \
    Consider that Budget is {budget} USD. Try to find accommodations in and around the Budget.",
    expected_output="All the details of a specifically chosen accommodation including the price, URL, and any image if available.",
    agent=hotel_reviewer,
)

# Create the Crew
travel_agent_crew = Crew(
    agents=[travel_expert, hotel_reviewer],
    tasks=[hotel_search_task, hotel_review_task],
    verbose=True,
    process=Process.sequential,
    # step_callback=[conversation_logger]
)

event_criteria = {
    'location': 'Japan, Tokyo',
    'check_in': '23rd August, 2025',
    'check_out': '25th August, 2025',
    'number_of_people': 2,
    'budget': 60
    }

result = travel_agent_crew.kickoff(inputs=event_criteria)
print(result)