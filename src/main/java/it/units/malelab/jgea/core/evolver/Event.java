/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package it.units.malelab.jgea.core.evolver;

import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.order.PartiallyOrderedCollection;

import java.io.Serializable;
import java.util.HashMap;
import java.util.Map;

/**
 * @author eric
 */
public class Event<G, S, F> implements Serializable {

  private final Evolver.State state;
  private final PartiallyOrderedCollection<Individual<G, S, F>> orderedPopulation;
  private final PartiallyOrderedCollection<Individual<G, S, F>> serializationPop;
  private final Map<String, Object> attributes;

  public Event(Evolver.State state, PartiallyOrderedCollection<Individual<G, S, F>> orderedPopulation, PartiallyOrderedCollection<Individual<G, S, F>> serializationPop) {
    this.state = state.copy();
    this.orderedPopulation = orderedPopulation;
    this.serializationPop = serializationPop;
    attributes = new HashMap<>();
  }

  public Event(Evolver.State state, PartiallyOrderedCollection<Individual<G, S, F>> orderedPopulation){
    this(state,orderedPopulation,null);
  }

  public Evolver.State getState() {
    return state;
  }

  public PartiallyOrderedCollection<Individual<G, S, F>> getOrderedPopulation() {
    return orderedPopulation;
  }

  public Map<String, Object> getAttributes() {
    return attributes;
  }

  public PartiallyOrderedCollection<Individual<G, S, F>> getSerializationPop(){
    return this.serializationPop;
  }

}
