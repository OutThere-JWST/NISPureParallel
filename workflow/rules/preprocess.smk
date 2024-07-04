# Preprocess Rule
rule preProcess:
    input:
        lambda wildcards: [f'RATE/{f.replace('uncal','rate')}' for f in uncal[wildcards.field]]
    output:
        'logs/{field}-proc.log'
    log:
        'logs/{field}-proc.log'
    group:
        lambda wildcards: f'proc-{groups[wildcards.field]}'
    conda:
        '../envs/grizli.yaml'
    resources:
        tasks = lambda wildcards: len(uncal[wildcards.field])
    shell:
        """
        ./workflow/scripts/preprocess.py {resources.tasks} &> {log}
        """