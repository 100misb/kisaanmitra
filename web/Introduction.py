import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import streamlit as st
from PIL import Image
from pathlib import Path

FILE_PATH = Path(__file__).resolve()
PROJECT_DIR = FILE_PATH.parents[0]
IMAGE_PATH = PROJECT_DIR / "introduction_1_image.jpg"

st.set_page_config(page_title="KisaanMitra", layout="wide")

@st.cache_resource
def load_image():
    image = Image.open(str(IMAGE_PATH))
    return image

with st.container() : 
    _, image_col, _ = st.columns([1,4,1])
    
    with image_col :
        st.image(load_image(), use_column_width=True)

st.markdown("**How many farmers are relying on Food Prices?**")

st.markdown(
    "India holds the record for second-largest agricultural land in the world, with around 58% of the Indian Population depends on agriculture for their livelihood."
)

st.markdown("**What bad price on crops leads to in Farmers life?**")

st.markdown(
    "Farmers face significant challenges when the crop prices fall too low (below the cost of production) and has a huge impact on their livelihoods. This can be challenging for small-scale farmer who completely rely on agriculture for their livelihoods. They are forced to sell their crops at very low price which leave them in a loss and struggle to meet the necessities like enough food, healthcare."
)

st.markdown("**Govt Initiatives in providing better MSP to farmers?**")

st.markdown(
    "The government has initiated few number schemes aiming to provide better Minimum Support Prices (MSP) to farmers which include providing subsidies for fertilizer and irrigation and investing in rural infrastructure. Government has implemented several measures to ensure that farmers receive better MSPs for their crops. One of the key initiatives is the Pradhan Mantri Annadata Aay SanraksHan Abhiyan (PM-AASHA), which was launched in 2018 to provide a solution for the MSP and procurement of crops. In addition, the government has increased the MSPs for various crops, including wheat, paddy, and pulses, and has also extended the procurement of crops to more states and regions."
)

st.markdown("**What is the Role of Retail Markets in Farmers?**")

st.markdown(
    "In addition to government support, Retail markets play a crucial role in connecting farmers with consumers and providing them an outlet to sell their produce directly to consumers which allows farmers to skip the intermediaries and receive a higher price for their goods."
)

st.markdown(
    "Retail markets can help to support local food systems and promote sustainable agriculture by encouraging the production and consumption of locally grown foods."
)
