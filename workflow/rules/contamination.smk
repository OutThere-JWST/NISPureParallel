# Contamination Rule
rule contam:
    input:
        'logs/{field}-mos.log'
    output:
        'logs/{field}-contam.log'
    conda:
        '../envs/grizli.yaml'
    log:
        'logs/{field}-contam.log'
    shell:
        """
        ./workflow/scripts/contamination.py {wildcards.field} --ncpu {threads} &> {log}
        """