all: install start

install:
	docker compose build

start:
	docker compose up

stop:
	docker compose down

restart:
	docker compose restart

logs:
	docker compose logs -f

clean: stop
		sudo docker system prune -af

re: clean
	make all

.PHONY: all install start stop restart logs clean re