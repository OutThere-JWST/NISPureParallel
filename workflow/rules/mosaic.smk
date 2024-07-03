# Mosaic Rule
rule mosaic:
    input:
        'FIELDS/{field}/logs/proc.out'
    output:
        'FIELDS/{field}/logs/mos.out'
    conda:'../envs/grizli.yaml'
    log:
        stdout='logs/{field}-mos.out',
        stderr='logs/{field}-mos.err'
    shell:
        """
        ./workflow/scripts/mosaic.py {wildcards.field} > {log.stdout} 2> {log.stderr}
        """