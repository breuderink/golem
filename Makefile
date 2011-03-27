TESTRUNNER=nosetests --with-coverage --with-doctest --cover-package=golem

.PHONY: all test

all: test 

dist:
	./setup.py sdist

test:
	$(TESTRUNNER) golem
