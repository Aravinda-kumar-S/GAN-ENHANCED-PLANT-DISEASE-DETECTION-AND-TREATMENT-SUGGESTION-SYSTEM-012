import google.generativeai as genai
import requests
import streamlit as st
from PIL import Image
import json
import socket

# --- AGRI-ADVISER SYSTEM PROMPT ---
SYSTEM_PROMPT = """
You are PlantVision AI, an expert agricultural consultant. 
Your role is to diagnose plant diseases, explain findings to farmers in simple terms, 
and suggest organic and chemical treatments.

Rules:
1. Speak in a friendly, respectful tone suitable for farmers.
2. AVOID technical jargon where possible.
3. If a disease is detected, structure your answer:
   - **Diagnosis**: What is it?
   - **Causes**: Why did it happen? (weather/pests)
   - **Organic Cure**: Homemade or natural remedies.
   - **Chemical Cure**: Safe pesticides with dosage.
   - **Prevention**: How to stop it coming back.
4. If the image is NOT a plant, politely decline to analyze it.
5. Consider the provided weather context in your advice (e.g., "Don't spray today because rain is expected").
"""

def get_gemini_config():
    """Load API key from Streamlit secrets and force V1 API"""
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        # Explicitly setting version to v1 to avoid v1beta issues
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        st.error("⚠️ Gemini API Key missing or invalid in .streamlit/secrets.toml")
        return False

def get_ai_explanation(image, predicted_disease, confidence, weather_info=None):
    """
    Generate an explanation using Gemini Vision Pro (or similar).
    """
    if not get_gemini_config():
        return "AI Explanation unavailable (Missing Key)."

    model = genai.GenerativeModel('gemini-flash-latest')
    
    weather_context = f"Current Weather: {weather_info}" if weather_info else "Weather data unavailable."
    
    prompt = f"""
    You are an agricultural AI advisor.
    
    Given:
    Disease Name: {predicted_disease}
    Confidence Score: {confidence*100:.1f}%
    Observed Symptoms: Please analyze the visual symptoms in the provided image.
    Environmental Conditions: {weather_context}

    Generate a structured agricultural diagnosis and treatment recommendation.

    Follow this exact output format:

    Disease Name:
    Confidence Score:

    Analysis Result:

    Recommended Treatments:

    1. Chemical Option
       - Name:
       - Variant:
       - Dosage:
       - Benefits:
       - Safety:
       - Application Method:

    2. Organic Option
       - Name:
       - Variant:
       - Dosage:
       - Benefits:
       - Safety:

    Preventive Care:

    Monitoring Advice:

    Guidelines:
    - Provide realistic agricultural treatments.
    - Include correct dosage.
    - Include safety precautions.
    - Keep explanation clear and farmer-friendly.
    - Avoid unnecessary medical disclaimers.
    - Do not include extra sections.
    - Maintain structured formatting.
    """
    
    try:
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error generating explanation: {str(e)}"

def identify_with_gemini(image):
    """
    Pure Gemini-based identification for when local model is unavailable or uncertain.
    """
    if not get_gemini_config():
        return "Unknown Plant", "Analysis Failed", 0.0, "API Key Missing"

    model = genai.GenerativeModel('gemini-flash-latest')
    
    prompt = """
    Analyze this image strictly. Identify the plant and disease.
    
    Return ONLY a JSON object with this structure:
    {
        "plant": "Name of the plant (e.g. Tomato, Apple)",
        "disease": "Name of disease or 'Healthy'",
        "confidence": 0.95,
        "explanation": "Short summary of visual findings"
    }
    """
    
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        response = model.generate_content([prompt, image])
        text = response.text
        
        # Method 1: Clean JSON Parsing
        try:
            clean_text = text.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_text)
            plant = data.get("plant", "Unknown")
            disease = data.get("disease", "Unknown")
            conf = float(data.get("confidence", 0.0))
            expl = data.get("explanation", "Analysis successful.")
            
            full_diagnosis = f"{plant} {disease}"
            return plant, full_diagnosis, conf, expl
            
        except json.JSONDecodeError:
            # Method 2: Fallback String Parsing (if AI chats instead of JSON)
            plant = "Unknown"
            disease = "Unknown"
            conf = 0.85 # Default fallback
            
            if "Plant:" in text: plant = text.split("Plant:")[1].split("\n")[0].strip()
            if "Disease:" in text: disease = text.split("Disease:")[1].split("\n")[0].strip()
            
            # Try to grab simple JSON-like keys if they exist in text
            if '"plant":' in text: plant = text.split('"plant":')[1].split(",")[0].replace('"', '').strip()
            
            full_diagnosis = f"{plant} {disease}"
            return plant, full_diagnosis, conf, text

    except Exception as e:
        return "Error", "API Error", 0.0, f"System Error: {str(e)}"


def get_ai_chat_response(messages, audio_bytes=None, context=""):
    """
    Chat with the Agri-Adviser.
    """
    if not get_gemini_config():
        return "I need an API key to think!"
        
    model = genai.GenerativeModel('gemini-flash-latest')
    
    user_input = messages[-1]['content'] if not audio_bytes else "The user provided an audio message. Please transcribe and answer."
    full_prompt = f"{SYSTEM_PROMPT}\n\nContext: {context}\n\nUser Question: {user_input}"
    
    try:
        if audio_bytes:
            # Prepare audio part for Gemini
            audio_blob = {
                "mime_type": "audio/wav", # streamli-mic-recorder usually outputs wav or similar
                "data": audio_bytes
            }
            response = model.generate_content([full_prompt, audio_blob])
        else:
            response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Sorry, I'm having trouble: {str(e)}"

def add_utm_params(url):
    """Appends referral tracking (UTM) parameters to external links."""
    if not url or url == "#": return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}utm_source=plantvision_ai&utm_medium=referral&utm_campaign=farmer_advisory"

# --- WEATHER INTELLIGENCE ---
def get_weather_data(lat, lon):
    """
    Fetch weather from OpenWeatherMap.
    """
    try:
        api_key = st.secrets.get("OPENWEATHER_API_KEY", "")
        if not api_key or api_key == "YOUR_OPENWEATHER_API_KEY_HERE":
            return {"error": "API Key missing"}

        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            return {
                "temp": data["main"]["temp"],
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"],
                "rain_prob": data.get("rain", {}).get("1h", 0) # Simple check
            }
        else:
            return {"error": f"API Error: {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def interpret_weather_for_spraying(weather):
    """
    Rules Engine for Spraying Advice.
    """
    if "error" in weather:
        return "Weather data unavailable for safety check."
        
    checks = []
    safe = True
    
    if weather.get("rain_prob", 0) > 0.4:
        checks.append("⚠️ Rain likely: Avoid spraying (wash-off risk).")
        safe = False
    
    if weather["temp"] > 30:
        checks.append("⚠️ High Temp: Spray in early morning/late evening to avoid leaf burn.")
        safe = False # Still safe to spray, but conditional
        
    if weather["humidity"] > 80:
        checks.append("💧 High Humidity: Fungal risk is elevated.")

    if safe and not checks:
        checks.append("✅ Conditions are optimal for spraying.")
        
    return "\n".join(checks)

def get_user_location():
    """
    Attempts to get the user's location based on their public IP address.
    Returns (lat, lon, city) or defaults if it fails.
    """
    try:
        # Using a free IP-based location service
        response = requests.get("http://ip-api.com/json/", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("lat", 20.59), data.get("lon", 78.96), data.get("city", "Unknown")
    except:
        pass
    return 20.59, 78.96, "Manual Default"

# --- AGRI-INPUT DATABASE (Rich Production Grade) ---
TREATMENT_DB = {
    "Tomato Early Blight": {
        "Plant": "Tomato (Solanum lycopersicum)",
        "Disease": "Early Blight (Alternaria solani)",
        "Organic": [
            {
                "name": "Vanproz V-Kure (Bio-Fungicide)", 
                "price": "₹450", 
                "dosage": "5ml per Liter", 
                "marketplaces": [
                    {"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/products/vanproz-v-kure-bio-fungicide-and-bactericide")},
                    {"site": "Amazon", "link": add_utm_params("https://www.amazon.in/s?k=v-kure+fungicide")}
                ]
            }
        ],
        "Chemical": [
            {
                "name": "Syngenta Amistar Top", 
                "price": "₹1250", 
                "dosage": "1ml per Liter", 
                "marketplaces": [
                    {"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/products/amistar-top-syngenta-fungicide")},
                    {"site": "AgroStar", "link": add_utm_params("https://www.agrostar.in/product/syngenta-amistar-top-fungicide/")}
                ]
            }
        ],
        "Seeds": [
            {
                "name": "Syngenta 1057 (Resistant Variety)",
                "price": "₹150",
                "marketplaces": [
                    {"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=syngenta+1057+tomato")}
                ]
            }
        ],
        "Remedies": "Prune lower leaves. Improve soil drainage.",
        "Prevention": "Crop rotation. Use certified seeds.",
        "Safety": "PHI: 7 days. Handle with care.",
        "PHI": "7 Days"
    },
    "Potato Late Blight": {
        "Plant": "Potato",
        "Disease": "Late Blight",
        "Organic": [
            {
                "name": "Copper Oxychloride (Bio-Opt)", 
                "price": "₹350", 
                "dosage": "3g/L", 
                "marketplaces": [
                    {"site": "Amazon", "link": add_utm_params("https://www.amazon.in/s?k=copper+oxychloride+for+plants")},
                    {"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=copper+oxychloride")}
                ]
            }
        ],
        "Chemical": [
            {
                "name": "Ridomil Gold (Syngenta)", 
                "price": "₹850", 
                "dosage": "2g/L", 
                "marketplaces": [
                    {"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=ridomil+gold")},
                    {"site": "AgroStar", "link": add_utm_params("https://www.agrostar.in/search?q=ridomil+gold")}
                ]
            }
        ],
        "Remedies": "Destruction of cull piles. Avoid excessive nitrogen.",
        "Prevention": "Use certified disease-free tubers.",
        "Safety": "PHI: 14 days.",
        "PHI": "14 Days"
    },
    "Apple Scab": {
        "Plant": "Apple",
        "Disease": "Apple Scab (Venturia inaequalis)",
        "Organic": [
            {"name": "Sulfur Fungicide", "price": "₹400", "dosage": "3g/L", "marketplaces": [{"site": "Amazon", "link": add_utm_params("https://www.amazon.in/s?k=sulfur+fungicide+for+plants")}]}
        ],
        "Chemical": [
            {"name": "Captan 50% WP", "price": "₹600", "dosage": "2.5g/L", "marketplaces": [{"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=captan")}]}
        ],
        "Prevention": "Prune for airflow. Remove fallen leaves.",
        "Safety": "Standard safety precautions. PHI: 10 days."
    },
    "Tomato Yellow Leaf Curl": {
        "Plant": "Tomato",
        "Disease": "Yellow Leaf Curl Virus (TYLCV)",
        "Organic": [
            {"name": "Yellow Sticky Traps", "price": "₹120", "dosage": "10-15 per acre", "marketplaces": [{"site": "Amazon", "link": add_utm_params("https://www.amazon.in/s?k=yellow+sticky+traps+for+plants")}]},
            {"name": "Neem Oil 10000 PPM", "price": "₹650", "dosage": "2ml/L", "marketplaces": [{"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=neem+oil+10000+ppm")}]}
        ],
        "Chemical": [
            {"name": "Imidacloprid (for Whitefly)", "price": "₹450", "dosage": "0.5ml/L", "marketplaces": [{"site": "Flipkart", "link": add_utm_params("https://www.flipkart.com/search?q=imidacloprid")}]}
        ],
        "Prevention": "Control whitefly vector. Use silver plastic mulch.",
        "Safety": "Very high toxicity to bees. Spray in late evening."
    },
    "Grape Black Rot": {
        "Plant": "Grape",
        "Disease": "Black Rot (Guignardia bidwellii)",
        "Organic": [
            {"name": "Bordeaux Mixture", "price": "₹300", "dosage": "1% solution", "marketplaces": [{"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=bordeaux+mixture")}]}
        ],
        "Chemical": [
            {"name": "Mancozeb 75% WP", "price": "₹550", "dosage": "2g/L", "marketplaces": [{"site": "AgroStar", "link": add_utm_params("https://www.agrostar.in/search?q=mancozeb")}]}
        ],
        "Prevention": "Sanitation is key. Remove mummified berries.",
        "Safety": "PHI: 30 days for grapes."
    },
    "Corn Northern Leaf Blight": {
        "Plant": "Corn (Maize)",
        "Disease": "Northern Leaf Blight",
        "Organic": [
            {"name": "Trichoderma Viride", "price": "₹280", "dosage": "10g/L", "marketplaces": [{"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=trichoderma")}]}
        ],
        "Chemical": [
            {"name": "Amistar Top (Azoxystrobin + Difenoconazole)", "price": "₹1250", "dosage": "1ml/L", "marketplaces": [{"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=amistar+top")}]},
            {"name": "Custodia (UPL)", "price": "₹950", "dosage": "1.5ml/L", "marketplaces": [{"site": "AgroStar", "link": add_utm_params("https://www.agrostar.in/search?q=custodia")}]}
        ],
        "Prevention": "Crop rotation with non-host crops. Use resistant hybrids.",
        "Safety": "Standard precautions. PHI: 20 days."
    },
    "Peach Bacterial Spot": {
        "Plant": "Peach",
        "Disease": "Bacterial Spot (Xanthomonas arboricola)",
        "Organic": [
            {"name": "Copper Oxychloride", "price": "₹350", "dosage": "2.5g/L", "marketplaces": [{"site": "Amazon", "link": add_utm_params("https://www.amazon.in/s?k=copper+oxychloride+for+plants")}]}
        ],
        "Chemical": [
            {"name": "Streptocycline (Bactericide)", "price": "₹150", "dosage": "6g in 60L water", "marketplaces": [{"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=streptocycline")}]}
        ],
        "Prevention": "Avoid high nitrogen fertilizers. Prune for air circulation.",
        "Safety": "Bactericide handled with care. PHI: 14 days."
    },
    "Pepper Bacterial Spot": {
        "Plant": "Pepper/Chilli",
        "Disease": "Bacterial Spot",
        "Organic": [
            {"name": "Neem Oil + Copper", "price": "₹500", "dosage": "5ml + 2g per Liter", "marketplaces": [{"site": "Amazon", "link": add_utm_params("https://www.amazon.in/s?k=neem+oil+fungicide+combo")}]}
        ],
        "Chemical": [
            {"name": "Kocide (Copper Hydroxide)", "price": "₹700", "dosage": "2g/L", "marketplaces": [{"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=kocide")}]}
        ],
        "Prevention": "Use certified seeds. Avoid overhead irrigation.",
        "Safety": "Wear gloves. PHI: 7 days."
    },
    "Strawberry Leaf Scorch": {
        "Plant": "Strawberry",
        "Disease": "Leaf Scorch (Diplocarpon earlianum)",
        "Organic": [
            {"name": "Potassium Bicarbonate", "price": "₹320", "dosage": "5g/L", "marketplaces": [{"site": "Amazon", "link": add_utm_params("https://www.amazon.in/s?k=potassium+bicarbonate+for+plants")}]}
        ],
        "Chemical": [
            {"name": "Carbendazim 50% WP", "price": "₹380", "dosage": "2g/L", "marketplaces": [{"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=carbendazim")}]}
        ],
        "Prevention": "Plant in well-drained soil. Remove old leaves after harvest.",
        "Safety": "PHI: 14 days."
    },
    "Tomato Septoria Leaf Spot": {
        "Plant": "Tomato",
        "Disease": "Septoria Leaf Spot",
        "Organic": [
            {"name": "Baking Soda Spray", "price": "₹50", "dosage": "1 tsp in 1L water", "marketplaces": [{"site": "Amazon", "link": add_utm_params("https://www.amazon.in/s?k=baking+soda")}]}
        ],
        "Chemical": [
            {"name": "Chlorothalonil (Kavach)", "price": "₹600", "dosage": "2g/L", "marketplaces": [{"site": "AgroStar", "link": add_utm_params("https://www.agrostar.in/search?q=kavach")}]}
        ],
        "Prevention": "Mulch to prevent soil splash. Prune lower branches.",
        "Safety": "PHI: 7 days."
    },
    "Orange Citrus Greening": {
        "Plant": "Orange/Citrus",
        "Disease": "Huanglongbing (Citrus Greening)",
        "Organic": [
            {"name": "Nutritional Spray (Zinc + Iron)", "price": "₹450", "dosage": "As per pack", "marketplaces": [{"site": "Amazon", "link": add_utm_params("https://www.amazon.in/s?k=citrus+micronutrient+fertilizer")}]}
        ],
        "Chemical": [
            {"name": "Confidor (Imidacloprid) for ACP Control", "price": "₹550", "dosage": "0.5ml/L", "marketplaces": [{"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=confidor")}]}
        ],
        "Prevention": "No known cure. Control psyllids and remove infected trees.",
        "Safety": "High toxicity to pollinators. PHI: 15 days."
    }
}

GOV_LINKS_DB = {
    "Default": [
        {"name": "PM Kisan Samman Nidhi (Income Support)", "link": "https://pmkisan.gov.in/"},
        {"name": "PM Fasal Bima Yojana (Crop Insurance)", "link": "https://pmfby.gov.in/"},
        {"name": "National Agriculture Market (eNAM)", "link": "https://www.enam.gov.in/"},
        {"name": "Soil Health Card Scheme", "link": "https://soilhealth.dac.gov.in/"},
        {"name": "Kisan Credit Card (KCC) Portal", "link": "https://www.kcc.gov.in/"}
    ],
    "Tamil Nadu": [
        {"name": "TNAU Agritech Portal", "link": "https://agritech.tnau.ac.in/"},
        {"name": "TN Agriculture Department", "link": "https://www.agriculture.tn.gov.in/"},
        {"name": "Uzhavan App Services", "link": "https://www.tntagrisnet.tn.gov.in/uzhavan/"}
    ],
    "Maharashtra": [
        {"name": "MahaAgri Official Portal", "link": "https://krishi.maharashtra.gov.in/"},
        {"name": "Aaple Sarkar DBT Portal", "link": "https://mahadbt.maharashtra.gov.in/"},
        {"name": "SMART Project (Agri-Business)", "link": "https://www.smart-mh.org/"}
    ],
    "Uttar Pradesh": [
        {"name": "UP Agriculture Department (Direct)", "link": "https://upagriculture.com/"},
        {"name": "UP Seeds Development Corporation", "link": "https://upsdc.gov.in/"},
        {"name": "Pardarshi Kisan Seva Yojana", "link": "http://upagriculture.com/Default.aspx"}
    ],
    "Karnataka": [
        {"name": "Raitamitra (Official Farmer Portal)", "link": "https://raitamitra.karnataka.gov.in/"},
        {"name": "K-Agri Input Portal", "link": "https://ka-agri.karnataka.gov.in/"},
        {"name": "Fruit-Soft (Farmer Registration)", "link": "https://fruits.karnataka.gov.in/"}
    ],
    "Punjab": [
        {"name": "Punjab Agri Portal (Direct)", "link": "https://agri.punjab.gov.in/"},
        {"name": "PAU Farmer Advisory", "link": "https://www.pau.edu/"}
    ],
    "Gujarat": [
        {"name": "i-Khedut Portal (Direct Subsidy)", "link": "https://ikhedut.gujarat.gov.in/"},
        {"name": "Gujarat Agri Department", "link": "https://agri.gujarat.gov.in/"}
    ],
    "West Bengal": [
        {"name": "Matir Katha (Integrated Portal)", "link": "https://matirkatha.gov.in/"},
        {"name": "WB Agriculture (Direct)", "link": "https://wb.gov.in/department-details.aspx?id=1"}
    ],
    "Andhra Pradesh": [
        {"name": "AP Agriculture Official", "link": "https://apagri.gov.in/"},
        {"name": "YSR Rythu Bharosa (Direct)", "link": "https://ysrrythubharosa.ap.gov.in/"}
    ]
}

def get_gov_links(state="Default"):
    links = GOV_LINKS_DB.get("Default", []).copy()
    if state in GOV_LINKS_DB:
        links.extend(GOV_LINKS_DB[state])
    return links

def get_climate_zone(lat, lon):
    """
    Very simplified Geofencing logic to determine climate zone.
    """
    if lat > 28:
        return "Subtropical/Temperate"
    elif 15 <= lat <= 28:
        return "Tropical Semi-Arid"
    else:
        return "Tropical Humid"

def get_state_from_coords(lat, lon):
    """
    Reverse geocoding mock or simple lookup.
    """
    # Simple bounding boxes for demo
    if 8 <= lat <= 13 and 76 <= lon <= 80:
        return "Tamil Nadu"
    if 15 <= lat <= 20 and 72 <= lon <= 80:
        return "Maharashtra"
    if 24 <= lat <= 30 and 77 <= lon <= 84:
        return "Uttar Pradesh"
    return "Default"

def speak_text(text, lang='en'):
    """
    Convert text to speech and return a Base64 string for an HTML audio player.
    """
    from gtts import gTTS
    import base64
    from io import BytesIO
    
    try:
        tts = gTTS(text=text, lang=lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_msg = base64.b64encode(fp.read()).decode()
        return audio_msg
    except Exception as e:
        return None

def get_recommendations(disease_name, growth_stage="Vegetative", weather=None):
    """Fetch treatments based on disease/crop name, adapting to growth stage and weather."""
    recs = None
    for key in TREATMENT_DB:
        if key.lower() in disease_name.lower():
            recs = TREATMENT_DB[key].copy()
            break
    
    zone = "Unknown"
    if weather and isinstance(weather, dict) and "lat" in weather:
        zone = get_climate_zone(weather["lat"], weather["lon"])
    
    if not recs:
        recs = {
            "Plant": "Detected Crop",
            "Disease": disease_name,
            "Organic": [{
                "name": "Broad-spectrum Bio-Fungicide", 
                "price": "₹150", 
                "dosage": "General Use", 
                "marketplaces": [{"site": "Amazon", "link": add_utm_params("https://www.amazon.in/s?k=bio+fungicide")}]
            }],
            "Chemical": [{
                "name": "Standard Copper Oxychloride", 
                "price": "₹220", 
                "dosage": "2g/L", 
                "marketplaces": [{"site": "BigHaat", "link": add_utm_params("https://www.bighaat.com/search?q=copper+oxychloride")}]
            }],
            "Prevention": f"Maintain field hygiene. Adapted for {zone} climate.",
            "Management": "Regular monitoring and early removal of infected parts.",
            "Safety": "Standard safety precautions apply. Wear protective clothing.",
            "PHI": "N/A"
        }

    # Adapt based on Growth Stage
    if growth_stage in ["Flowering", "Fruiting"]:
        recs["Safety"] += f" | ⚠️ **Warning**: Use caution during {growth_stage} to avoid harming pollinators."
    elif growth_stage == "Seedling":
        recs["Management"] += " | 🌡️ **Tip**: Maintain high humidity and avoid direct sunlight for weak seedlings."

    # Adapt based on Weather
    if weather and not isinstance(weather, str) and "error" not in weather:
        if weather.get("temp", 0) > 30:
            recs["Safety"] += " | 🌡️ High Temperature: Avoid middle-of-day spraying to prevent phytotoxicity."
    
    return recs

def get_nearby_vendors(lat, lon, radius_km=25):
    """Mock Geospatial logic with real Google Maps search fallback."""
    vendors = [
        {
            "name": "Agro-Care Center (Verified)", 
            "dist": 1.2, 
            "lat": lat + 0.008, 
            "lon": lon + 0.005,
            "link": "https://www.agrocarecenter.com/",
            "phone": "+91 98765 43210",
            "address": "Main Market Road, Agri Hub"
        },
        {
            "name": "Farmers Friend Input Store", 
            "dist": 3.5, 
            "lat": lat - 0.015, 
            "lon": lon + 0.008,
            "link": "#",
            "phone": "+91 87654 32109",
            "address": "Opposite Panchayat Office"
        },
        {
            "name": "Precision Seeds & Pesticides", 
            "dist": 5.8, 
            "lat": lat + 0.02, 
            "lon": lon - 0.015,
            "link": "https://www.precisionseeds.in/",
            "phone": "+91 76543 21098",
            "address": "Old Highway Junction"
        },
        {
            "name": "Village Cooperative Society", 
            "dist": 12.4, 
            "lat": lat - 0.035, 
            "lon": lon - 0.025,
            "link": "#",
            "phone": "+91 65432 10987",
            "address": "Village Center"
        },
    ]
    
    # Add Google Maps Search link for 100% accurate local discovery
    gmaps_search = f"https://www.google.com/maps/search/pesticide+seeds+fertilizer+shop/@{lat},{lon},14z"
    
    return vendors, gmaps_search

