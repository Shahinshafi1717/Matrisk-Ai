# fix_app.py
# Fixes the syntax error in app.py

with open('src/dashboard/app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the problematic line
old = 'st.metric("Bollinger Z", f"{latest["bollinger_z"]:.2f}")'
new = 'bz = latest["bollinger_z"]\n        st.metric("Bollinger Z", f"{bz:.2f}")'
content = content.replace(old, new)

with open('src/dashboard/app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed! Now run: streamlit run src/dashboard/app.py")