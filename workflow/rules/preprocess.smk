# Preprocess Rule
rule preProcess:
    input:
        lambda wildcards: [f'RATE/{f.replace('uncal','rate')}' for f in uncal[wildcards.field]]
    output:
        'logs/{field}-proc.log'
    conda:
        '../envs/grizli.yaml'
    log:
        'logs/{field}-proc.log'
    shell:
        """
        ./workflow/scripts/preprocess.py {wildcards.field} &> {log}
        """