# Mosaic Rule
rule mosaic:
    input:
        'FIELDS/{field}/logs/proc.log'
    output:
        'FIELDS/{field}/logs/mos.log'
    log:
        'FIELDS/{field}/logs/mos.log'
    group:
        lambda wildcards: f'mos-{groups[wildcards.field]}'
    # resources:
    #     tasks = lambda wildcards: len(uncal[wildcards.field])
    shell:
        """
        pixi run --no-lockfile-update --environment grizli ./workflow/scripts/mosaic.py {wildcards.field} > {log} 2>&1
        """