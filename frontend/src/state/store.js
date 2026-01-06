class Store {
  constructor() {
    this.state = {
      active_pane: "train",
      data_loaded: { train: false, classify: false },
      active_bundle_id: null,
    };
    this.listeners = [];
  }

  subscribe(listener) {
    this.listeners.push(listener);
    return () => {
      this.listeners = this.listeners.filter((l) => l !== listener);
    };
  }

  setState(nextState) {
    this.state = nextState;
    this.listeners.forEach((listener) => listener(this.state));
  }

  getState() {
    return this.state;
  }
}

export const store = new Store();
