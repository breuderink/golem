TESTRUNNER=nosetests --with-coverage --with-doctest --cover-package=golem

.PHONY: all test

all: test 

test:
	$(TESTRUNNER) golem
