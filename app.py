import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load API key from secrets
try:
    api_key = st.secrets["GROQ_API_KEY"]
    llm = ChatOpenAI(
        model="llama3-70b-8192",
        openai_api_key=api_key,
        openai_api_base="https://api.groq.com/openai/v1",
        temperature=0.5
    )
except:
    llm = None

# Datacrumbs information
@st.cache_data
def get_datacrumbs_info():
    return """
    DATACRUMBS - COMPLETE INFORMATION

    LOCATION & CONTACT:
    Address: Room # 105, Shahrah-e-Faisal, Karachi, Pakistan
    Phone: +92 336 250 7273
    Email: help@datacrumbs.org
    Website: datacrumbs.org

    MENTOR & TEAM:
    Lead Mentor: Abis Hussain Syed
    Expert instructors with industry experience

    COURSES & PRICING:
    • Data Science Bootcamp - Rs. 29,999 (4 months)
    • Data Analytics Bootcamp - Rs. 29,999 (4 months) 
    • Business Intelligence Bootcamp - Rs. 29,999 (4 months)
    • GenAI Bootcamp (Generative AI) - Rs. 29,999
    • Ultimate Python Bootcamp - Rs. 25,000
    • SQL Zero to Hero - Rs. 15,000
    • Excel for Everyone - Rs. 12,000

    FEATURES:
    ✓ Industry-ready curriculum
    ✓ Hands-on projects
    ✓ Certification upon completion
    ✓ Internship opportunities
    ✓ Career placement assistance
    ✓ Live mentorship sessions
    ✓ 24/7 community support
    """

# Page config
st.set_page_config(page_title="Datacrumbs Chatbot", page_icon="💬")

# Simple header
st.title("Datacrumbs Chatbot")
st.subheader("I'm your virtual assistant today")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content=f"""
You are a helpful sales assistant for Datacrumbs, an educational platform in Karachi, Pakistan.

Answer questions about courses, pricing, location, and services professionally and friendly.

DATACRUMBS INFO:
{get_datacrumbs_info()}

Keep responses conversational and helpful.
""")
    ]

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, SystemMessage):
        continue
    
    role = "assistant" if isinstance(message, AIMessage) else "user"
    with st.chat_message(role):
        st.write(message.content)

# Chat input
if prompt := st.chat_input("💬 Your question here..."):
    # Add user message
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        if llm:
            try:
                response = llm(st.session_state.messages)
                st.write(response.content)
                st.session_state.messages.append(AIMessage(content=response.content))
            except:
                fallback_msg = "Please visit datacrumbs.org or contact us at help@datacrumbs.org for more information."
                st.write(fallback_msg)
                st.session_state.messages.append(AIMessage(content=fallback_msg))
        else:
            fallback_msg = "Please visit datacrumbs.org or contact us at help@datacrumbs.org for more information."
            st.write(fallback_msg)
            st.session_state.messages.append(AIMessage(content=fallback_msg))

# Enrollment Form
st.markdown("---")
st.subheader("📝 Enroll Now")

with st.form("enrollment_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        name = st.text_input("Full Name *", placeholder="Enter your full name")
        email = st.text_input("Email *", placeholder="your.email@example.com")
        phone = st.text_input("Phone Number *", placeholder="+92 300 1234567")
    
    with col2:
        course = st.selectbox("Course of Interest *", [
            "Select a course...",
            "Data Science Bootcamp - Rs. 29,999",
            "Data Analytics Bootcamp - Rs. 29,999", 
            "Business Intelligence Bootcamp - Rs. 29,999",
            "GenAI Bootcamp - Rs. 29,999",
            "Ultimate Python Bootcamp - Rs. 25,000",
            "SQL Zero to Hero - Rs. 15,000",
            "Excel for Everyone - Rs. 12,000"
        ])
        
        experience = st.selectbox("Programming Experience", [
            "No experience",
            "Beginner (some basics)",
            "Intermediate",
            "Advanced"
        ])
        
        education = st.selectbox("Education Level", [
            "High School",
            "Bachelor's Degree",
            "Master's Degree",
            "Other"
        ])
    
    message = st.text_area("Additional Message (Optional)", 
                          placeholder="Tell us about your goals or any questions...")
    
    submitted = st.form_submit_button("Submit Enrollment Request", type="primary")
    
    payment_option = st.radio("Payment Preference", [
        "Full Payment (One-time)",
        "Installment Plan (Monthly)"
    ])
    
    if submitted:
        if name and email and phone and course != "Select a course...":
            # Extract course fee
            course_fee = course.split(" - Rs. ")[1] if " - Rs. " in course else "Contact for pricing"
            
            st.success("✅ Thank you! Your enrollment request has been submitted.")
            
            # Auto-scroll to bottom after form submission
            st.markdown("""
            <script>
                setTimeout(function() {
                    window.scrollTo(0, document.body.scrollHeight);
                }, 100);
            </script>
            """, unsafe_allow_html=True)
            
            # Mini enrollment document
            st.markdown("""
            ---
            ### 📋 ENROLLMENT CONFIRMATION DOCUMENT
            """)
            
            st.info(f"""
            **STUDENT INFORMATION:**
            - **Name:** {name}
            - **Email:** {email}
            - **Phone:** {phone}
            - **Course:** {course.split(' - Rs.')[0]}
            - **Experience Level:** {experience}
            - **Education:** {education}
            
            **COURSE FEE INFORMATION:**
            - **Total Course Fee:** {course_fee}
            - **Payment Option:** {payment_option}
            - **Payment Details:** {"Pay full amount upfront" if payment_option == "Full Payment (One-time)" else "Monthly installments available - discuss with our team"}
            
            **BANK ACCOUNT FOR FEE PAYMENT:**
            - **Bank Name:** [Bank Name]
            - **Account Title:** Datacrumbs
            - **Account Number:** [Account Number]
            - **IBAN:** [IBAN Number]
            - **Branch Code:** [Branch Code]
            
            ⚠️ **Important:** Please send payment confirmation (screenshot/receipt) to help@datacrumbs.org after making the payment.
            
            **CONTACT & LOCATION:**
            - **Phone:** +92 336 250 7273
            - **Email:** help@datacrumbs.org
            - **Address:** Room # 105, Shahrah-e-Faisal, Karachi, Pakistan
            - **Website:** datacrumbs.org
            
            **NEXT STEPS:**
            Our team will contact you within 24 hours to discuss:
            - Course schedule and start date
            - Payment plan details (if installment selected)
            - Required materials and setup
            - Welcome session booking
            """)
            
            st.success("💾 **Please save this information for your records!**")
            
        else:
            st.error("❌ Please fill in all required fields marked with *")

# Simple footer
st.markdown("---")
st.markdown("**Contact:** help@datacrumbs.org | +92 336 250 7273 | [datacrumbs.org](https://datacrumbs.org)")
