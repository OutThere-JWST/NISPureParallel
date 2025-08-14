# # High memory fields
# high_memory_fields = ['gru-00']

# # Compute resources for all fields
# cpus_per_task = workflow.resource_settings.default_resources.parsed['cpus_per_task']
# resources = {
#     field: max(1, cpus_per_task // 2) if field in high_memory_fields else cpus_per_task
#     for field in FIELDS
# }


# Extraction Rule
rule extract:
    input:
        'FIELDS/{field}/logs/contam.log'
    output:
        'FIELDS/{field}/logs/extr.log'
    log:
        'FIELDS/{field}/logs/extr.log'
    # resources:
    #     cpus_per_task = lambda wc: resources[wc.field]
    shell:
        """
        pixi run --no-lockfile-update --environment grizli ./workflow/scripts/extract.py {wildcards.field} --ncpu {resources.cpus_per_task} > {log} 2>&1
        """