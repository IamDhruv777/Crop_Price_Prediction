from pathlib import Path

guard = """
# Authentication & Role Guard: Only Researchers can access advanced analytics pages
if not st.session_state.get("logged_in", False) or st.session_state.get("user_role") != "Researcher":
    st.switch_page("app.py")
"""

pages_dir = Path("dashboard/pages")
for p in pages_dir.glob("*.py"):
    content = p.read_text(encoding="utf-8")
    lines = content.splitlines()
    new_lines = []
    inserted = False
    
    # Remove any old guard lines first
    filtered_lines = []
    skip = False
    for line in lines:
        if 'if not st.session_state.get("logged_in"' in line:
            skip = True
            continue
        if skip and 'st.switch_page("app.py")' in line:
            skip = False
            continue
        filtered_lines.append(line)
        
    for line in filtered_lines:
        new_lines.append(line)
        if "st.set_page_config" in line and not inserted:
            new_lines.append(guard)
            inserted = True
            
    p.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"Updated guard in {p.name}")
