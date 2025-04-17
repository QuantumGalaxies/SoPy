from sqlalchemy import create_engine

def conn():
    server    = 'jonathan.jerke-sierranevada.nord'
    database  = 'papers'
    return create_engine('postgresql://{}/{}'.format(server,database))

