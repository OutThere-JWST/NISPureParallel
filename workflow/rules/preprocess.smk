# Preprocess Rule
rule preProcess:
    input:
        lambda wildcards: [f'RATE/{f.replace('uncal','rate')}' for f in uncal[wildcards.field]]
    output:
        'FIELDS/{field}/logs/proc.out'
    conda:'../envs/grizli.yaml'
    log:
        stdout='logs/{field}-proc.out',
        stderr='logs/{field}-proc.err'
    shell:
        """
        ./workflow/scripts/preprocess.py {wildcards.field} > {log.stdout} 2> {log.stderr}
        """