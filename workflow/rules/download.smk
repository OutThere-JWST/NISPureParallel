# Download Rule
def create_rule(field):
    rule:
        name: f"download_{field}"
        output:
            [f'FIELDS/{field}/UNCAL/{f}' for f in uncal[field]]
        log: 
            f'FIELDS/{field}/logs/download.log'
        group:
            f'download-{groups[field]}'
        resources:
        shell: 
            f"""
            mkdir -p FIELDS/{field}/logs
            ./workflow/scripts/download.py {field} --ncpu {{resources.cpus_per_task}} > {{log}} 2>&1
            pixi run --no-lockfile-update --environment jwst crds sync --contexts $CRDS_CONTEXT --fetch-references --dataset-files FIELDS/{field}/UNCAL/* >> {{log}} 2>&1
            """

# Create rules for all fields
for field in FIELDS: create_rule(field)