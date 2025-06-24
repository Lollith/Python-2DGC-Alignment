DEV_COMPOSE=docker-compose.dev.yml
PROD_COMPOSE=docker-compose.yml


all:  start_prod 

all_dev: build_dev start_dev 

# build_nist:
# 	@if [ -z "$$(docker images -q nist)" ]; then \
# 		echo "Image Docker NIST non trouvée. Construction en cours..."; \
# 		docker compose -f $(DEV_COMPOSE) build nist; \
# 	else \
# 		echo "Image Docker NIST déjà construite."; \
# 	fi


build_dev: 
	docker compose -f $(DEV_COMPOSE) build 2dgc_id

rebuild_dev: 
	docker compose -f $(DEV_COMPOSE) build --no-cache


# check_image:
# 	@if [ -z "$$(docker images -q 2dgc_id-app)" ]; then \
# 		echo "Image Docker non trouvée. Construction en cours..."; \
# 		docker compose -f $(PROD_COMPOSE) build; \
# 	else \
# 		echo "Image Docker déjà construite."; \
# 	fi

start_dev : 
	docker compose -f $(DEV_COMPOSE) up

# start_prod: check_image
start_prod:
	docker compose -f $(PROD_COMPOSE) up 

stop:
	- docker compose -f $(DEV_COMPOSE) down -v
	- docker compose -f $(PROD_COMPOSE) down -v

logs_dev:
	docker compose -f $(DEV_COMPOSE) logs -f

logs_prod:
	docker compose -f $(PROD_COMPOSE) logs -f

rebuild_prod:
	docker compose -f $(PROD_COMPOSE) build --no-cache
	echo "Image Docker reconstruite avec succès."

clean: stop
	-sudo docker compose -f $(PROD_COMPOSE) rm -sfv 2dgc_id
	-sudo docker compose -f $(DEV_COMPOSE) rm -sfv 2dgc_id
	-sudo docker rmi python-2dgc-alignment-2dgc_id || true
	-sudo docker volume ls -qf "dangling=true" | xargs -r sudo docker volume rm
	-sudo docker image prune -f
	echo "Conteneurs, réseaux, images, volumes des environnements PROD et DEV supprimés."

clean_build: stop
	-sudo docker builder prune -f

clean_nist: stop
	-sudo docker compose -f $(PROD_COMPOSE) rm -sfv nist
	-sudo docker compose -f $(DEV_COMPOSE) rm -sfv nist
	-sudo docker rmi python-2dgc-alignment-nist || true
	echo "Conteneurs, réseaux, images, volumes de l'environnement NIST supprimés."

re_dev: clean
	make all_dev

re: clean
	make all

.PHONY: all all_dev build_dev start_dev start_prod stop logs_dev logs_prod rebuild_prod clean clean_nist re re_dev
