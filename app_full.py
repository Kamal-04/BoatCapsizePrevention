import streamlit as st
import math
import plotly.graph_objects as go
from llama_index.llms.openai import OpenAI
from llama_index.core.storage import StorageContext
from llama_index.core.response.pprint_utils import pprint_response
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, load_index_from_storage, ServiceContext
from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core import set_global_service_context
import os
import constant
import requests

# Set API keys
os.environ['OPENAI_API_KEY'] = constant.OPENAI_API_KEY_PAID
os.environ['GROQ_API_KEY'] = constant.GROQ_API_KEY

# Heel Test Simulator Code
boat_types = {
    "Pontoon Boat": {
        "weight": 1800,
        "length": 7,
        "breadth": 2.5,
        "extreme_breadth": 2.7,
        "passenger_capacity": 10,
        "heel_threshold": 10
    },
    "Small Cabin Cruiser": {
        "weight": 3000,
        "length": 9,
        "breadth": 3,
        "extreme_breadth": 3.2,
        "passenger_capacity": 6,
        "heel_threshold": 20
    },
    "Day Cruiser": {
        "weight": 1500,
        "length": 6,
        "breadth": 2.3,
        "extreme_breadth": 2.5,
        "passenger_capacity": 8,
        "heel_threshold": 25
    },
    "Fishing Boat": {
        "weight": 2000,
        "length": 6.5,
        "breadth": 2.4,
        "extreme_breadth": 2.6,
        "passenger_capacity": 6,
        "heel_threshold": 30
    }
}

def get_weather_data(city):
    API_KEY = constant.OPENWEATHER_API_KEY
    BASE_URL = 'https://api.openweathermap.org/data/2.5/weather?'
    url = BASE_URL + "appid=" + API_KEY + "&q=" + city
    response = requests.get(url).json()
    return response

def calculate_heel_test(data, weather_data):
    boat_weight = data['boatWeight']
    passenger_count = data['passengerCount']
    passenger_weight = data['passengerWeight']
    crew_weight = data['crewWeight']
    tank_status = data['tankStatus']
    extreme_breadth = data['extremeBreadth']
    distance_ws1 = data['distanceWS1']
    distance_ws2 = data['distanceWS2']
    water_density = data['waterDensity']
    weight_ws1 = data['weightWS1']
    weight_ws2 = data['weightWS2']

    total_passenger_weight = passenger_count * passenger_weight
    total_crew_weight = crew_weight
    total_weight = boat_weight + total_passenger_weight + total_crew_weight

    heeling_moment_ws1 = weight_ws1 * distance_ws1
    heeling_moment_ws2 = weight_ws2 * distance_ws2
    heeling_moment = heeling_moment_ws1 + heeling_moment_ws2

    gz = 1  # Righting lever, assumed value
    angle_of_heel = math.degrees(math.atan(heeling_moment / (extreme_breadth * total_weight * gz)))

    # Weather adjustment
    wind_speed = weather_data['wind']['speed']
    visibility = weather_data['visibility']

    # Adjust the heel threshold based on weather
    base_threshold = boat_types[data['boat_type']]['heel_threshold']
    weather_adjusted_threshold = base_threshold - (wind_speed * 0.5)  # Adjusting the threshold based on weather
    weather_adjusted_threshold = max(weather_adjusted_threshold, 5)  

    return {
        'heeling_moment': heeling_moment,
        'angle_of_heel': angle_of_heel,
        'weather_adjusted_threshold': weather_adjusted_threshold
    }

def visualize_heel(angle_of_heel, weather_data):
    fig = go.Figure()

    # Water surface vis
    fig.add_shape(type="rect", x0=-2, y0=-0.5, x1=2, y1=0,
                  fillcolor="lightblue", line_color="blue")

    # Boat vis
    boat_x = [-0.5, 0.5, 0]
    boat_y = [0, 0, 1]
    rotated_x, rotated_y = rotate_points(boat_x, boat_y, angle_of_heel)
    fig.add_trace(go.Scatter(x=rotated_x, y=rotated_y, fill="toself", fillcolor="brown", line_color="brown"))

    # Wind vis
    wind_speed = weather_data['wind']['speed']
    wind_dir = weather_data['wind']['deg']
    wind_x = math.cos(math.radians(wind_dir))
    wind_y = math.sin(math.radians(wind_dir))
    fig.add_annotation(x=1.5, y=1, ax=1.5 + wind_x, ay=1 + wind_y,
                       arrowhead=2, arrowsize=1, arrowwidth=2,
                       text=f"Wind: {wind_speed} m/s")

    # Indicating visibilly
    visibility = weather_data['visibility']
    fig.add_annotation(x=-1.5, y=1.5, text=f"Visibility: {visibility/1000:.1f} km",
                       showarrow=False)

    fig.update_layout(
        xaxis_range=[-2, 2],
        yaxis_range=[-0.5, 2],
        xaxis_title="",
        yaxis_title="",
        showlegend=False,
        height=500,
        width=700
    )

    return fig

def rotate_points(x, y, angle_degrees):
    angle_radians = math.radians(angle_degrees)
    rotated_x = []
    rotated_y = []
    for xi, yi in zip(x, y):
        rotated_x.append(xi * math.cos(angle_radians) - yi * math.sin(angle_radians))
        rotated_y.append(xi * math.sin(angle_radians) + yi * math.cos(angle_radians))
    return rotated_x, rotated_y

#Chatbot
storage_path = "./vectorstore"
documents_path = "./document"

llm = Groq(model="llama3-8b-8192", api_key=constant.GROQ_API_KEY)
service_context = ServiceContext.from_defaults(llm=llm)
Settings.embed_model = OpenAIEmbedding()
set_global_service_context(service_context)

@st.cache_resource(show_spinner=False)
def initialize_chatbot():
    if not os.path.exists(storage_path):
        documents = SimpleDirectoryReader("D:\ChatBot\LIL_Final\data_base").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=storage_path)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
    return index

def clear_chat_history():
    st.session_state['messages'] = [
        {'role': 'assistant', 'content': "Hello, Ask your queries"}
    ]

#App
st.set_page_config(page_title="Marine Tools", page_icon="🚢", layout="wide")

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Heel Test Simulator", "Marine Chatbot"])

    if page == "Home":
        st.title("Welcome to Marine Tools")
        st.write("This application provides two main features:")
        st.write("1. Heel Test Simulator: Simulate and visualize boat heel angles with real-time weather integration.")
        st.write("2. Marine Chatbot: Ask questions about the maritime industry and the Indian Navy.")
        st.write("Use the sidebar to navigate between these tools.")

    elif page == "Heel Test Simulator":
        st.title("Heel Test Simulation with Weather Integration")
        
        #location
        location = st.text_input("Enter your location (city name):", "Chennai")
        
        #fetching weather
        weather_data = get_weather_data(location)
        if weather_data.get('cod') != 200:
            st.error(f"Error fetching weather data: {weather_data.get('message', 'Unknown error')}")
        else:
            st.success(f"Weather data fetched successfully for {location}")

        selected_boat_type = st.selectbox("Select Boat Type", list(boat_types.keys()))
        boat_data = boat_types[selected_boat_type]

        col1, col2 = st.columns(2)
        with col1:
            boat_weight = st.number_input("Boat Weight (kg)", value=boat_data["weight"])
            boat_length = st.number_input("Boat Length (m)", value=boat_data["length"])
            boat_breadth = st.number_input("Boat Breadth (m)", value=boat_data["breadth"])
            extreme_breadth = st.number_input("Extreme Breadth (m)", value=boat_data["extreme_breadth"])
            passenger_count = st.number_input("Number of Passengers", value=0, max_value=boat_data["passenger_capacity"], step=1)
            passenger_weight = st.number_input("Average Passenger Weight (kg)", value=75.0)
            crew_weight = st.number_input("Average Crew Weight (kg)", value=80.0)

        with col2:
            tank_status = st.slider("Tank Fill Level (%)", 0, 100, 50)
            liquid_type = st.selectbox("Type of Liquid", ["Fresh Water", "Sea Water", "Fuel"])
            total_weight = st.number_input("Total Weight to be Placed on Board (kg)", value=500.0)
            weight_ws1 = st.number_input("Weight WS1 (kg)", value=250.0)
            weight_ws2 = st.number_input("Weight WS2 (kg)", value=250.0)
            distance_ws1 = st.number_input("Distance for Shifting Weight WS1 (m)", value=1.5)
            distance_ws2 = st.number_input("Distance for Shifting Weight WS2 (m)", value=1.5)
            water_density = st.number_input("Water Density", value=1025.0)

        if st.button("Simulate Test"):
            data = {
                'boat_type': selected_boat_type,
                'boatWeight': boat_weight,
                'passengerCount': passenger_count,
                'passengerWeight': passenger_weight,
                'crewWeight': crew_weight,
                'tankStatus': tank_status,
                'extremeBreadth': extreme_breadth,
                'distanceWS1': distance_ws1,
                'distanceWS2': distance_ws2,
                'waterDensity': water_density,
                'weightWS1': weight_ws1,
                'weightWS2': weight_ws2
            }
            result = calculate_heel_test(data, weather_data)
            
            st.subheader("Simulation Results")
            st.write(f"Heeling Moment: {result['heeling_moment']:.2f} N·m")
            st.write(f"Angle of Heel: {result['angle_of_heel']:.2f} degrees")
            st.write(f"Weather-Adjusted Heel Threshold: {result['weather_adjusted_threshold']:.2f} degrees")

            st.subheader("Heel Visualization with Weather Impact")
            fig = visualize_heel(result['angle_of_heel'], weather_data)
            st.plotly_chart(fig)

            if result['angle_of_heel'] > result['weather_adjusted_threshold']:
                st.warning(f"Warning: The heel angle ({result['angle_of_heel']:.2f}°) exceeds the weather-adjusted threshold ({result['weather_adjusted_threshold']:.2f}°)!")
            else:
                st.success(f"The boat appears to be stable. Heel angle ({result['angle_of_heel']:.2f}°) is within the weather-adjusted threshold ({result['weather_adjusted_threshold']:.2f}°).")

            #Displaying weather
            st.subheader("Current Weather Conditions")
            st.write(f"Temperature: {weather_data['main']['temp'] - 273.15:.1f}°C")
            st.write(f"Wind Speed: {weather_data['wind']['speed']} m/s")
            st.write(f"Wind Direction: {weather_data['wind']['deg']}°")
            st.write(f"Visibility: {weather_data['visibility']/1000:.1f} km")
            st.write(f"Description: {weather_data['weather'][0]['description']}")

        st.info("Note: These simulations are based on simplified models and real-time weather data. Always consult with naval architects and follow proper safety regulations.")

    elif page == "Marine Chatbot":
        st.title("Chat to the Marine Bot")
        st.sidebar.button("Clear chat History", on_click=clear_chat_history)
        
        index = initialize_chatbot()
        memory = ChatMemoryBuffer.from_defaults(token_limit=540000)

        if "messages" not in st.session_state.keys():
            st.session_state['messages'] = [
                {"role": "assistant", "content": "Ask me a question !"}
            ]

        chat_engine = index.as_chat_engine(chat_mode="condense_question",
                                           memory=memory,
                                           context_prompt=(
                                               "You are a chatbot, able to have normal interactions, as well as talk"
                                               " about the Maritime industry and the Indian Navy."
                                               "Here are the relevant documents for the context:\n"
                                               "{context_str}"
                                               "\nInstruction: Use the previous chat history, or the context above, to interact and help the user."
                                           ),
                                           llm=OpenAI(temperature=0.8),
                                           verbose=True)

        if prompt := st.chat_input("Your question"):
            st.session_state.messages.append({"role": "user", "content": prompt})

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = chat_engine.chat(prompt)
                    st.write(response.response)
                    message = {"role": "assistant", "content": response.response}
                    st.session_state.messages.append(message)

if __name__ == "__main__":
    main()