import streamlit as st
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np


# ==============================
# PAGE CONFIGURATION
# ==============================

st.set_page_config(
    page_title="Hotel Booking Prediction",
    page_icon="🏨",
    layout="wide"
)


# ==============================
# LOAD MODEL AND FILES
# ==============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "hotel_booking_ann_model.keras"
    )


@st.cache_resource
def load_scaler():
    return joblib.load(
        "hotel_booking_scaler.pkl"
    )


@st.cache_resource
def load_features():
    return joblib.load(
        "hotel_booking_features.pkl"
    )


model = load_model()
scaler = load_scaler()
feature_names = load_features()


# ==============================
# TITLE
# ==============================

st.title("🏨 Hotel Booking Cancellation Prediction")

st.write(
    "ANN-based Hotel Booking Prediction System"
)

st.success("Model loaded successfully!")


# ==============================
# MODEL INFORMATION
# ==============================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "Artificial Neural Network"
    )

with col2:
    st.metric(
        "Features",
        len(feature_names)
    )

with col3:
    st.metric(
        "Status",
        "Ready"
    )


st.divider()


# ==============================
# BOOKING FORM
# ==============================

st.header("📝 Booking Details")

col1, col2, col3 = st.columns(3)


# ------------------------------
# Column 1
# ------------------------------

with col1:

    lead_time = st.number_input(
        "Lead Time",
        min_value=0,
        max_value=1000,
        value=100
    )

    arrival_year = st.selectbox(
        "Arrival Year",
        [2015, 2016, 2017],
        index=2
    )

    arrival_month = st.selectbox(
        "Arrival Month",
        [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December"
        ]
    )

    arrival_week = st.number_input(
        "Arrival Week Number",
        min_value=1,
        max_value=53,
        value=30
    )

    arrival_day = st.number_input(
        "Arrival Day",
        min_value=1,
        max_value=31,
        value=15
    )


# ------------------------------
# Column 2
# ------------------------------

with col2:

    weekend_nights = st.number_input(
        "Weekend Nights",
        min_value=0,
        max_value=20,
        value=1
    )

    week_nights = st.number_input(
        "Week Nights",
        min_value=0,
        max_value=50,
        value=3
    )

    adults = st.number_input(
        "Adults",
        min_value=1,
        max_value=10,
        value=2
    )

    children = st.number_input(
        "Children",
        min_value=0,
        max_value=10,
        value=0
    )

    babies = st.number_input(
        "Babies",
        min_value=0,
        max_value=10,
        value=0
    )


# ------------------------------
# Column 3
# ------------------------------

with col3:

    meal = st.selectbox(
        "Meal",
        ["BB", "HB", "SC", "FB", "Undefined"]
    )

    market_segment = st.selectbox(
        "Market Segment",
        [
            "Online TA",
            "Offline TA/TO",
            "Groups",
            "Direct",
            "Corporate",
            "Complementary",
            "Aviation",
            "Undefined"
        ]
    )

    distribution_channel = st.selectbox(
        "Distribution Channel",
        [
            "TA/TO",
            "Direct",
            "Corporate",
            "GDS",
            "Undefined"
        ]
    )

    deposit_type = st.selectbox(
        "Deposit Type",
        [
            "No Deposit",
            "Non Refund",
            "Refundable"
        ]
    )

    customer_type = st.selectbox(
        "Customer Type",
        [
            "Transient",
            "Transient-Party",
            "Contract",
            "Group"
        ]
    )


st.divider()


# ==============================
# ROOM DETAILS
# ==============================

st.header("🛏️ Room & Guest Details")

col1, col2, col3 = st.columns(3)


with col1:

    reserved_room = st.selectbox(
        "Reserved Room Type",
        list("ABCDEFGH")
    )

    assigned_room = st.selectbox(
        "Assigned Room Type",
        list("ABCDEFGH")
    )

    is_repeated_guest = st.selectbox(
        "Repeated Guest",
        [0, 1]
    )


with col2:

    booking_changes = st.number_input(
        "Booking Changes",
        min_value=0,
        max_value=50,
        value=0
    )

    previous_cancellations = st.number_input(
        "Previous Cancellations",
        min_value=0,
        max_value=50,
        value=0
    )

    previous_bookings = st.number_input(
        "Previous Bookings Not Cancelled",
        min_value=0,
        max_value=50,
        value=0
    )


with col3:

    adr = st.number_input(
        "ADR",
        min_value=0.0,
        max_value=10000.0,
        value=100.0
    )

    parking_spaces = st.number_input(
        "Required Parking Spaces",
        min_value=0,
        max_value=10,
        value=0
    )

    special_requests = st.number_input(
        "Special Requests",
        min_value=0,
        max_value=10,
        value=0
    )


st.divider()


# ==============================
# EXTRA DETAILS
# ==============================

st.header("📊 Additional Details")

col1, col2, col3 = st.columns(3)


with col1:

    country = st.text_input(
        "Country",
        value="PRT"
    )

    agent = st.number_input(
        "Agent",
        min_value=0,
        value=9
    )


with col2:

    company = st.number_input(
        "Company",
        min_value=0,
        value=0
    )

    waiting_list = st.number_input(
        "Days in Waiting List",
        min_value=0,
        max_value=500,
        value=0
    )


with col3:

    hotel = st.selectbox(
        "Hotel",
        [
            "Resort Hotel",
            "City Hotel"
        ]
    )


# ==============================
# PREDICTION BUTTON
# ==============================

st.divider()

predict_button = st.button(
    "🔮 PREDICT BOOKING",
    use_container_width=True
)


if predict_button:

    st.info("Preparing prediction...")

    # --------------------------
    # CREATE INPUT DATA
    # --------------------------

    total_nights = (
        weekend_nights +
        week_nights
    )

    total_guests = (
        adults +
        children +
        babies
    )

    room_changed = int(
        reserved_room != assigned_room
    )


    input_data = {

        "lead_time": lead_time,

        "arrival_date_year": arrival_year,

        "arrival_date_month": arrival_month,

        "arrival_date_week_number": arrival_week,

        "arrival_date_day_of_month": arrival_day,

        "stays_in_weekend_nights":
            weekend_nights,

        "stays_in_week_nights":
            week_nights,

        "adults": adults,

        "children": children,

        "babies": babies,

        "meal": meal,

        "country": country,

        "market_segment": market_segment,

        "distribution_channel":
            distribution_channel,

        "is_repeated_guest":
            is_repeated_guest,

        "previous_cancellations":
            previous_cancellations,

        "previous_bookings_not_canceled":
            previous_bookings,

        "reserved_room_type":
            reserved_room,

        "assigned_room_type":
            assigned_room,

        "booking_changes":
            booking_changes,

        "deposit_type":
            deposit_type,

        "agent": agent,

        "company": company,

        "days_in_waiting_list":
            waiting_list,

        "customer_type":
            customer_type,

        "adr": adr,

        "required_car_parking_spaces":
            parking_spaces,

        "total_of_special_requests":
            special_requests,

        "total_nights":
            total_nights,

        "total_guests":
            total_guests,

        "room_changed":
            room_changed
    }


    # --------------------------
    # DATAFRAME
    # --------------------------

    input_df = pd.DataFrame(
        [input_data]
    )


    # --------------------------
    # ONE-HOT ENCODING
    # --------------------------

    input_df = pd.get_dummies(
        input_df,
        drop_first=True
    )


    # --------------------------
    # MATCH TRAINING FEATURES
    # --------------------------

    input_df = input_df.reindex(
        columns=feature_names,
        fill_value=0
    )


    # --------------------------
    # CONVERT DATA TYPE
    # --------------------------

    input_df = input_df.astype(int)


    # --------------------------
    # SCALE DATA
    # --------------------------

    input_scaled = scaler.transform(
        input_df
    )


    # --------------------------
    # MODEL PREDICTION
    # --------------------------

    probability = model.predict(
        input_scaled,
        verbose=0
    )[0][0]


    probability_percent = (
        probability * 100
    )


    prediction = (
        "CANCELLED"
        if probability >= 0.5
        else "NOT CANCELLED"
    )


    # ==========================
    # RESULT
    # ==========================

    st.divider()

    st.header("🎯 Prediction Result")


    col1, col2 = st.columns(2)


    with col1:

        st.metric(
            "Cancellation Probability",
            f"{probability_percent:.2f}%"
        )


    with col2:

        st.metric(
            "Prediction",
            prediction
        )


    st.progress(
        float(probability)
    )


    if probability >= 0.5:

        st.error(
            f"❌ Booking is likely to be CANCELLED"
        )

    else:

        st.success(
            f"✅ Booking is likely to NOT be CANCELLED"
        )