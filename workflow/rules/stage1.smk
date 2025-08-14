# Download Rule
rule stage1:
    input:
        'FIELDS/{field}/UNCAL/{file}_uncal.fits'
    output:
        'FIELDS/{field}/RATE/{file}_rate.fits'
    log: 
        'FIELDS/{field}/logs/files/{file}.log'
    group:
        'stage1'
    # resources:
    #     mem_mb = lambda _, input: 5 * input.size_mb
    shell: 
        """
        pixi run --no-lockfile-update --environment jwst ./workflow/scripts/stage1.py --scratch {input} {output} >> {log} 2>&1
        """