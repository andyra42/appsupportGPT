import requests
import streamlit as st

st.title("Input your Query")
selector = st.text_area(label="Input Query", value="", height=None, max_chars=None, key=None, help=None, on_change=None,
                        args=None, kwargs=None, placeholder=None, disabled=False, label_visibility="hidden")
query_text_sql = selector

if st.button("Search", type='primary'):
    data = requests.get("http://localhost:5110/api/prompt_route", params={"user_prompt": query_text_sql})
    data = data.json()
    print(data)
    #output_query = data["response"]
    st.header("Query Response")
    st.write(data)
