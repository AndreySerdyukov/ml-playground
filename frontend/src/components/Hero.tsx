// Хиро-секция главной в стиле Apple: крупный заголовок + сабхед.
export function Hero() {
  return (
    <section className="px-6 pt-20 pb-12 text-center sm:pt-24">
      <h1 className="mx-auto max-w-3xl text-4xl font-semibold leading-[1.05] tracking-tight text-ink sm:text-6xl">
        Machine learning, hands-on.
      </h1>
      <p className="mx-auto mt-5 max-w-xl text-lg text-slate sm:text-2xl">
        Pick a model, enter the inputs, get a prediction — right in your browser.
      </p>
    </section>
  );
}
