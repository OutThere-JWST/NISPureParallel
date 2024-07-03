# Contamination Rule
rule contam:
    input:
        'FIELDS/{field}/logs/mos.out'
    output:
        'FIELDS/{field}/logs/contam.out'
    conda:'../envs/grizli.yaml'
    log:
        stdout='logs/{field}-contam.out',
        stderr='logs/{field}-contam.err'
    shell:
        """
        ./workflow/scripts/contamination.py {wildcards.field} --ncpu {threads} > {log.stdout} 2> {log.stderr}
        """