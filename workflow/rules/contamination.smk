# Contamination Rule
rule contam:
    input:
        'logs/{field}-mos.log'
    output:
        'logs/{field}-contam.log'
    log:
        'logs/{field}-contam.log'
    group:
        lambda wildcards: f'contam-{groups[wildcards.field]}'
    conda:
        '../envs/grizli.yaml'
    shell:
        """
        ./workflow/scripts/contamination.py {wildcards.field} --ncpu {threads} &> {log}
        """