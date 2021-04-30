import sys

def title(text):
  if sys.platform.startswith('linux'):
    print('\033[1;32m\n{}'.format(text.upper()))
    print('-'*len(text), end='\n\n\033[m')
  else:
    print('\n{}'.format(text.upper()))
    print('-'*len(text), end='\n\n')


def header(text):
  print()
  if sys.platform.startswith('linux'):
    print('\033[1;31m-\033[0;34m=\033[1;31m-\033[0m' * 20)
    print('\033[1;31m{:^60}\033[m'.format(text))
    print('\033[1;31m-\033[0;34m=\033[1;31m-\033[0m' * 20)
  else:
    print('-=-' * 20)
    print('{:^60}'.format(text))
    print('-=-' * 20)

  print()


def highlight(text):
  if sys.platform.startswith('linux'):
    return '\033[1;32m{}\033[0m'.format(text)
  else:
    return text


def done():
  if sys.platform.startswith('linux'):
    print('\033[1;33mFeito!\033[0m\n')
  else:
    print('Feito!\n')


def print_distances_and_classes(distances_and_classes):
  i = 1
  for distance, classification in distances_and_classes:
    print('{}. {:.2f} ({})'.format(highlight(i), distance, highlight(classification)))
    i += 1


def print_votes(votes):
  for k, v in votes.items():
    print('{0}: {1} ocorrência{2} encontrada{2}'.format(highlight(k), v, 's' if v > 1 else ''))
