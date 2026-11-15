const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
const animations = [...document.querySelectorAll('.animation')];

function setAnimation(figure, playing) {
  const image = figure.querySelector('.demo-image');
  const button = figure.querySelector('.animation-toggle');
  const source = playing ? image.dataset.gif : image.dataset.poster;
  if (image.getAttribute('src') !== source) image.src = source;
  figure.dataset.playing = String(playing);
  button.setAttribute('aria-pressed', String(playing));
  button.querySelector('span').textContent = playing ? 'Show still frame' : 'Play GIF';
  button.querySelector('img').src = playing ? 'assets/icons/image.svg' : 'assets/icons/play.svg';
}

const observer = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;
    if (!motionPreference.matches && entry.target.dataset.userControlled !== 'true') {
      setAnimation(entry.target, true);
    }
    observer.unobserve(entry.target);
  });
}, { threshold: .1 });

animations.forEach(figure => {
  observer.observe(figure);
  figure.querySelector('button').addEventListener('click', () => {
    figure.dataset.userControlled = 'true';
    setAnimation(figure, figure.dataset.playing !== 'true');
  });
});

const dialog = document.querySelector('#framework-dialog');
const zoom = document.querySelector('.framework-zoom');
zoom.addEventListener('click', () => {
  dialog.showModal();
  document.body.style.overflow = 'hidden';
});
dialog.querySelector('button').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
dialog.addEventListener('close', () => {
  document.body.style.overflow = '';
  zoom.focus({ preventScroll: true });
});

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const text = document.querySelector('#citation').textContent;
  const status = document.querySelector('.copy-status');
  try {
    if (navigator.clipboard && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
    } else {
      const field = document.createElement('textarea');
      field.value = text;
      field.style.position = 'fixed';
      field.style.opacity = '0';
      document.body.append(field);
      field.select();
      const copied = document.execCommand('copy');
      field.remove();
      if (!copied) throw new Error('Clipboard unavailable');
    }
    status.textContent = 'BibTeX copied.';
  } catch {
    status.textContent = 'Select the citation text to copy it.';
  }
});
