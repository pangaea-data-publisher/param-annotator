#!/bin/sh

cd `dirname $0`
exec /usr/bin/daemon -o daemon.info -r -n param-annotator -i -D "$PWD" -- \
 python3 api/appv1.py config/termconf_production.ini
