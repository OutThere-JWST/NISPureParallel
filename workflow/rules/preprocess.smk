# Preprocess Rule
rule preProcess:
    input:
        lambda wildcards: [f'FIELDS/{wildcards.field}/RATE/{f.replace('uncal','rate')}' for f in uncal[wildcards.field]]
    output:
        'FIELDS/{field}/logs/proc.log'
    log:
        'FIELDS/{field}/logs/proc.log'
    group:
        lambda wildcards: f'proc-{groups[wildcards.field]}'
    conda:
        '../envs/grizli.yaml'
    # resources:
    #     tasks = lambda wildcards: len(uncal[wildcards.field])
    shell:
        """
        ./workflow/scripts/preprocess.py {wildcards.field} --ncpu {resources.cpus_per_task} > {log} 2>&1
        """