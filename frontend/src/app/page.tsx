import { PredictionWorkspace } from "@/components/PredictionWorkspace";

export default function Home() {
  return (
    <>
      <a className="skip-link" href="#prediction-workspace">
        Skip to customer assessment
      </a>

      <header className="site-header">
        <div className="shell site-header__inner">
          <div className="brand" aria-label="ChurnSignal home">
            <span className="brand__mark" aria-hidden="true">
              <span />
              <span />
              <span />
            </span>
            <span>
              Churn<span>Signal</span>
            </span>
          </div>
          <p className="site-header__meta">
            <span aria-hidden="true" />
            Telecom retention intelligence
          </p>
        </div>
      </header>

      <main>
        <section className="hero">
          <div className="shell hero__grid">
            <div className="hero__copy">
              <p className="eyebrow">Customer retention workspace</p>
              <h1>See churn risk before a customer walks away.</h1>
              <p className="hero__lede">
                Turn account and service signals into an immediate,
                model-powered risk assessment your retention team can act on.
              </p>
            </div>

            <dl className="hero__metrics" aria-label="Prediction model details">
              <div>
                <dt>Inputs</dt>
                <dd>19</dd>
                <p>customer signals</p>
              </div>
              <div>
                <dt>Decision point</dt>
                <dd>50%</dd>
                <p>churn threshold</p>
              </div>
              <div>
                <dt>Engine</dt>
                <dd>ML</dd>
                <p>saved pipeline</p>
              </div>
            </dl>
          </div>
        </section>

        <div className="shell" id="prediction-workspace">
          <PredictionWorkspace />
        </div>
      </main>

      <footer className="site-footer">
        <div className="shell site-footer__inner">
          <p>ChurnSignal · Decision support for telecom retention teams</p>
          <p>Predictions are estimates and should be reviewed in context.</p>
        </div>
      </footer>
    </>
  );
}
