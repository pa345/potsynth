/*
 * error.c
 *
 * Error handling, modeled on the GSL error handler: all library functions
 * report errors through potsynth_error(), which calls the current error
 * handler. Unlike GSL, the default handler only prints a message to stderr
 * and does not abort, since potsynth may be running inside another program
 * (e.g. Python).
 *
 * Copyright (C) 2026 Patrick Alken and Johnathan Burchill, BSD 3-Clause License
 */

#include <stdio.h>
#include <stdlib.h>

#include "potsynth_internal.h"

static void default_handler(const char * reason, const char * file, int line, int potsynth_errno);
static void no_handler(const char * reason, const char * file, int line, int potsynth_errno);

static potsynth_error_handler_t * current_handler = default_handler;

/*
potsynth_error()
  Report an error through the current error handler

Inputs: reason         - description of the error
        file           - source file where the error occurred
        line           - source line
        potsynth_errno - error code (POTSYNTH_Exxx)
*/

void
potsynth_error(const char * reason, const char * file, int line, int potsynth_errno)
{
  potsynth_error_handler_t * handler = current_handler;

  if (handler)
    (*handler)(reason, file, line, potsynth_errno);
}

/*
potsynth_set_error_handler()
  Set a new error handler

Inputs: new_handler - new handler; NULL disables error reporting

Return: previous handler
*/

potsynth_error_handler_t *
potsynth_set_error_handler(potsynth_error_handler_t * new_handler)
{
  potsynth_error_handler_t * previous = current_handler;

  current_handler = new_handler;

  return previous;
}

/*
potsynth_set_error_handler_off()
  Disable error reporting (functions still return error codes or NULL)

Return: previous handler
*/

potsynth_error_handler_t *
potsynth_set_error_handler_off(void)
{
  return potsynth_set_error_handler(no_handler);
}

/*
potsynth_strerror()
  Return a description of an error code

Inputs: potsynth_errno - error code (POTSYNTH_xxx)
*/

const char *
potsynth_strerror(const int potsynth_errno)
{
  switch (potsynth_errno)
    {
      case POTSYNTH_SUCCESS:
        return "success";

      case POTSYNTH_EINVAL:
        return "invalid argument";

      case POTSYNTH_ENOMEM:
        return "memory allocation failed";

      case POTSYNTH_EDOM:
        return "evaluation point outside the valid domain";

      case POTSYNTH_EUNSUP:
        return "unsupported option";

      default:
        return "unknown error";
    }
}

/* print the error to stderr (does not abort) */
static void
default_handler(const char * reason, const char * file, int line, int potsynth_errno)
{
  fprintf(stderr, "potsynth: %s:%d: ERROR: %s (%s)\n", file, line, reason,
          potsynth_strerror(potsynth_errno));
}

static void
no_handler(const char * reason, const char * file, int line, int potsynth_errno)
{
  (void) reason;
  (void) file;
  (void) line;
  (void) potsynth_errno;
}
