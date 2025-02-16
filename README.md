# ST1447-DewateringPredictor

## Running the project
    $ gh repo clone AndreaMalloni/ST1447-DewateringPredictor
    $ cd AndreaMalloni/ST1447-DewateringPredictor
    $ python3 -m venv ./venv
    $ pip install -r requirements.txt

# Introduction
This project focuses on analyzing data from a Cyber-Physical System (CPS) using Big Data Technologies. The CPS in question is a dewatering machine, specifically an industrial decanter centrifuge, which plays a crucial role in sludge dewatering across various industries such as wastewater treatment, food processing, and petrochemical sectors.

## Context and Motivation
Dewatering machines operate by separating solid particles from liquids using centrifugal force. The efficiency of this process depends on various factors, including drum and screw speeds, sludge inlet characteristics, and chemical dosing (e.g., polyelectrolytes). These machines are equipped with multiple sensors that monitor operational parameters, including temperature, vibration, turbidity, and moisture content. However, despite the availability of extensive real-time data, AI-driven analytics and Big Data approaches have not yet been fully leveraged to optimize the machine's performance.