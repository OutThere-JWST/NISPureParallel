# Preprocess Rule
rule preProcess:
    input:
        lambda wildcards: [f'RATE/{f.replace('uncal','rate')}' for f in uncal[wildcards.field]]
    output:
        'FIELDS/{field}/logs/proc.out'
    conda:'../envs/grizli.yaml'
    log:'logs/PreProcess_{field}.log'
    shell:
        """
        ./workflow/scripts/preprocess.py {wildcards.field}
        """