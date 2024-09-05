# Download Rule
def create_rule(field):
    rule:
        name: f"download_{field}"
        output:
            [f'UNCAL/{f}' for f in uncal[field]]
        log: 
            f'logs/{field}.download.log'
        group:
            f'download-{groups[field]}'
        shell: 
            f"""
            ./workflow/scripts/download.py {field} --ncpu {{resources.cpus_per_task}} > {{log}} 2>&1
            """

# Create rules for all fields
for field in FIELDS: create_rule(field)