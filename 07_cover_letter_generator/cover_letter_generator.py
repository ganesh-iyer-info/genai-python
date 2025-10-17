# Cover Letter Generator using Prompt Templates

resume_summary = """
Transformation leader and technical program manager with 15+ years of experience in cloud modernization, data center infrastructure, and enterprise delivery across financial, telecom, healthcare, and tech sectors. Skilled in Python, Agile, stakeholder alignment, and platform reliability.
"""

target_job_title = "Senior Technical Program Manager"
company_name = "Acme Corp"

# Prompt template
template = f"""
Dear Hiring Team at {company_name},

I am excited to apply for the {target_job_title} role. With a background as a transformation leader and technical program manager, I bring over 15 years of experience driving cloud modernization, data center infrastructure, and enterprise delivery across financial, telecom, healthcare, and tech sectors.

My strengths include aligning stakeholders, leading Agile teams, and ensuring platform reliability. I am proficient in Python and have a proven track record of managing complex programs that deliver strategic impact.

I am drawn to {company_name}'s commitment to innovation and excellence, and I would welcome the opportunity to contribute to your mission. Thank you for considering my application.

Sincerely,  
Ganesh Iyer
"""

# Output
print(template)